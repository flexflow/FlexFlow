/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "model.h"

using namespace Legion;

/* NOTE: This is the very, very un-performant.
- You should get rid of the functions and not do each sample
  individually
- You should at least use Eigen or uBlas or so for matrix multiplications.
  Preferably implement on GPU */

// Take exp pred with highest weight (debug)
void f_first(float* output,
            const float* gate_net_preds,
            float** exp_preds,
            int k,
            int out_dim)
{
  if(exp_preds[0] == 0) return;
  for(int i = 0; i < out_dim; i++) {
    output[i] = exp_preds[0][i];
  }
}

void f_first_back(float* output,
                  const float* gate_net_preds,
                  float** exp_preds,
                  int k,
                  int out_dim)
{
  for(int i = 0; i < out_dim; i++) {
    exp_preds[0][i] += output[i];
  }
}

// Multiply exp preds with gate weights (first loss in '91 Jacobs)
void f_mm(float* output,
          const float* gate_net_preds,
          float** exp_preds,
          int k,
          int out_dim) {
  // set output tensor to 0
  for(int i = 0; i < out_dim; i++) {
    output[i] = 0.0f;
  }

  // Multiply exp preds with weights
  for(int i = 0; i < k; i++) {
    if(exp_preds[i] == 0) continue; // dropped sample
    for(int j = 0; j < out_dim; j++) {
      output[j] += exp_preds[i][j]*gate_net_preds[i];
    }
  }
}


void f_mm_back(const float* output_grad,
              const float* gate_preds,
              float** exp_preds,
              float* gate_grads,
              float** exp_grads,
              int k,
              int out_dim)
{
  // gating net gradient
  for(int i = 0; i < k; i++) {
    if(exp_preds[i] == 0) continue; // dropped sample
    for(int j = 0; j < out_dim; j++) {
      gate_grads[i] += output_grad[j]*exp_preds[i][j];
    }
  }

  // expert gradients
  for(int i = 0; i < k; i++) {
    if(exp_preds[i] == 0) continue; // dropped sample
    for(int j = 0; j < out_dim; j++) {
      exp_grads[i][j] += gate_preds[i]*output_grad[j];
    }
  }
}


Tensor FFModel::aggregate(const Tensor* inputs, /* gate_preds, gate_assign, n * exp_pred */
                          int n, const char* name)
{
  Aggregate* aggr = new Aggregate(*this, inputs, n, name);
  layers.push_back(aggr);
  return aggr->outputs[0];
}


Aggregate::Aggregate(FFModel& model,
                    const Tensor* _inputs,
                    int _n, const char* name)
: Op(model, OP_AGGREGATE, name, _n+2/*num_inputs*/, 0/*num_weights*/, _inputs),
  n(_n)
  //profiling(model.config.profiling)
{
  assert(inputs[0]->num_dims == 2); // TODO: More flexible. pass in images etc.
  assert(inputs[1]->num_dims == 2);
  assert(inputs[0]->dims[0] == inputs[1]->dims[0]);
  assert(inputs[0]->dims[1] == inputs[1]->dims[1]);
  assert(n+2 == numInputs);
  assert(n > 0);

  int out_dim = inputs[2]->dims[0].size;
  int batch_size = inputs[0]->dims[1].size;
  outputs[0]->num_dims = 2;
  outputs[0]->dims[0].size = out_dim;
  outputs[0]->dims[1].size = batch_size;

  for(int i = 0; i < n; i++) {
    assert(inputs[i+2]->dims[0].size == out_dim);
  }

  numWeights = 0;
}

#ifdef DEADCODE
void Aggregate::create_weights(FFModel& model)
{
  // Do nothing
}


void Aggregate::create_output_and_partition(FFModel& model)
{
  // Retrieve the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);

  // Can only partition over the sample dim
  assert(part_rect.hi[0] == part_rect.lo[0]);

  int batch_size = inputs[0].adim[1];
  int out_dim = inputs[2].adim[0];

  const int dims[2] = {batch_size, out_dim};
  outputs[0] = model.create_tensor<2>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;


  // Compute partition bound for input
  for(int i = 0; i < n+2; i++) {
    Rect<2> input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[i]->part.get_index_partition());
    if (input_rect == part_rect) {
      input_lps[i] = inputs[i]->part;
      input_grad_lps[i] = inputs[i]->part_grad;
    } else {
      model.create_disjoint_partition<2>(
        inputs[i], (IndexSpaceT<2>)task_is, input_lps[i], input_grad_lps[i]);
    }
  }
}
#endif

OpMeta* Aggregate::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  return NULL;
}


/* TODO: ?? . Also produces warning [warning 1071] LEGION WARNING: Region
requirement 7 of operation Aggregate Init Task (UID 140) in parent task
top_level (UID 1) is using uninitialized data for field(s) 0 of logical
region (81,28,53) */
void Aggregate::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(AGGREGATE_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Aggregate)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i+2], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+2]->region));
    launcher.add_field(i+2, FID_DATA);
  }
  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(n+2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}


void aggregate_forward(float** exp_preds,
        const int* exp_assign,
        const float* gating_net_preds,
        float* output,
        int n, // num experts
        int k, // num chosen experts
        int exp_samples, // max samples per expert
        int batch_size,
        int out_dim)
{
  std::vector<int> expert_idx(n, 0);
  float* chosen_exp_preds[k];
  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < k; j++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[k*i + j];
      if(expert_idx[expert] >= exp_samples) {
        // dropped sample
        chosen_exp_preds[j] = 0;
        continue;
      }
      chosen_exp_preds[j] = exp_preds[expert] + expert_idx[expert]*out_dim;
      expert_idx[expert]++;
    }
    f_mm(output+i*out_dim, gating_net_preds+i*out_dim, chosen_exp_preds,
      k, out_dim);
  }
}


void aggregate_backward(float** exp_preds,
        float** exp_grads,
        const int* exp_assign,
        const float* gating_net_preds,
        float* gating_net_grads,
        float* output_grads,
        int n, // num experts
        int k, // num chosen experts
        int exp_samples, // max samples per expert
        int batch_size,
        int out_dim)
{
  std::vector<int> expert_idx(n, 0);
  float* chosen_exp_preds[k];
  float* chosen_exp_grads[k];
  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < k; j++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[k*i + j];
      if(expert_idx[expert] >= exp_samples) {
        // dropped sample
        chosen_exp_preds[j] = 0;
        chosen_exp_grads[j] = 0;
        continue;
      }
      chosen_exp_preds[j] = exp_preds[expert] + expert_idx[expert]*out_dim;
      chosen_exp_grads[j] = exp_grads[expert] + expert_idx[expert]*out_dim;
      expert_idx[expert]++;
    }
    f_mm_back(output_grads+i*out_dim, gating_net_preds+i*out_dim, chosen_exp_preds,
      gating_net_grads+i*out_dim, chosen_exp_grads, k, out_dim);
  }
}


void Aggregate::forward_task(const Task *task,
                             const std::vector<PhysicalRegion>& regions,
                             Context ctx, Runtime* runtime)
{
  int n = ((Aggregate*)task->args)->n;

  assert((int)regions.size() == n+3);
  assert((int)task->regions.size() == n+3);

  // get gate_pred, gate_assign, output
  const AccessorRO<float, 2> acc_gate_pred(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_gate_assign(regions[1], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[n+2], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[n+2].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  assert(rect_gate_pred.hi[0] - rect_gate_pred.lo[0] == rect_gate_assign.hi[0] - rect_gate_assign.lo[0]);
  assert(batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;

  // get exp_preds
  float* exp_preds[n];
  // get first exp_pred and row and out_dim
  Domain exp_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerWO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for(int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerWO<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  int k = (int)(rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1);

  aggregate_forward(exp_preds, acc_gate_assign.ptr(rect_gate_assign),
    acc_gate_pred.ptr(rect_gate_pred), acc_output.ptr(rect_output), n, k, rows,
    batch_size, out_dim);
}

void Aggregate::backward_task(const Task *task,
                              const std::vector<PhysicalRegion>& regions,
                              Context ctx, Runtime* runtime)
{
  int n = ((Aggregate*)task->args)->n;

  assert((int)regions.size() == 2*n+4);
  assert((int)task->regions.size() == 2*n+4);

  // get gate_pred, gate_grad, gate_assign, output_grad
  const AccessorRO<float, 2> acc_gate_pred(regions[0], FID_DATA);
  const AccessorRW<float, 2> acc_gate_grad(regions[1], FID_DATA);
  const AccessorRO<int, 2> acc_gate_assign(regions[2], FID_DATA);
  const AccessorRW<float, 2> acc_output_grad(regions[2*n+3], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_grad = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  Rect<2> rect_out_grad = runtime->get_index_space_domain(
      ctx, task->regions[2*n+3].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_grad.hi[1] - rect_gate_grad.lo[1] + 1);
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  assert(batch_size == rect_out_grad.hi[1] - rect_out_grad.lo[1] + 1);
  coord_t k = rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1;
  assert(rect_gate_pred.hi[0] - rect_gate_pred.lo[0] + 1 == k);
  assert(rect_gate_grad.hi[0] - rect_gate_grad.lo[0] + 1 == k);
  coord_t out_dim = rect_out_grad.hi[0] - rect_out_grad.lo[0] + 1;

  // get exp_preds
  float* exp_preds[n];
  // get first exp_pred and row
  Domain exp_domain = runtime->get_index_space_domain(
    ctx, task->regions[3].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerRW<float>(
    regions[3], task->regions[3], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for(int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+3].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerRW<float>(
      regions[i+3], task->regions[i+3], FID_DATA, ctx, runtime);

    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  // get chosen_exp_grads
  float* exp_grads[n];
  for(int i = 0; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[n+i+3].region.get_index_space());
    exp_grads[i] = helperGetTensorPointerRW<float>(
      regions[n+i+3], task->regions[n+i+3], FID_DATA, ctx, runtime);

    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  aggregate_backward(exp_preds, exp_grads, acc_gate_assign.ptr(rect_gate_assign),
    acc_gate_pred.ptr(rect_gate_pred), acc_gate_grad.ptr(rect_gate_grad),
    acc_output_grad.ptr(rect_out_grad), n, k, rows, batch_size, out_dim);
}


void Aggregate::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(AGGREGATE_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Aggregate)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i+2], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+2]->region));
    launcher.add_field(i+2, FID_DATA);
  }
  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(n+2, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}


void Aggregate::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(AGGREGATE_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Aggregate)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);

  // gate gradients
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);

  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(2, FID_DATA);

  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i+2], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+2]->region));
    launcher.add_field(i+3, FID_DATA);
  }

  // exp_preds gradients
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[i+2], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+2]->region_grad));
    launcher.add_field(i+n+3, FID_DATA);
  }

  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(2*n+3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}


bool Aggregate::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics)
{
  //TODO: implement
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.memory_requirement = 0;
  return false;
}

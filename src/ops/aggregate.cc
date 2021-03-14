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

// =========================================================
//              Start: Aggregation functions
//   Compute output given expert predictions and weights.
// =========================================================

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
  // TODO: dropped samples
  for(int i = 0; i < out_dim; i++) {
    exp_preds[0][i] += output[i];
  }


  // First one gets gradient.
  // Rest 0

  // One weight multiplied
}

// Multiply exp preds with gate weights
/* NOTE: These matrix multiplications could be sped up significantly.
Use uBlas or Eigen or so. Probably also worth it to implement for GPU */
void f_mm() {
  // Multiply exp preds with weights
}


void f_mm_back() {
  // part out / part exp_pred
  // Muliply input with gate_pred^T
  // Split up to experts according to assignment


  // part out / part gate_pred
  // Multiply input with exp_pred^T
  // Write to inputs[0]
  // TODO: Check if topK correct with setting gradient entries to 0
}

// =========================================================
//                End: Aggregation functions
// =========================================================


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
: Op(model, OP_AGGREGATE, name, _n+2, _inputs),
  n(_n)
  //profiling(model.config.profiling)
{
  assert(inputs[0].numDim == 2); // TODO: More flexible. pass in images etc.
  assert(inputs[1].numDim == 2);
  assert(inputs[0].adim[0] == inputs[1].adim[0]);
  assert(inputs[0].adim[1] == inputs[1].adim[1]);
  assert(n+2 == numInputs);
  assert(n > 0);

  int out_dim = inputs[2].adim[0];
  int batch_size = inputs[0].adim[1];
  outputs[0].numDim = 2;
  outputs[0].adim[0] = out_dim;
  outputs[0].adim[1] = batch_size;

  for(int i = 0; i < n; i++) {
    assert(inputs[i+2].adim[0] == out_dim);
  }

  numWeights = 0;
}

void Aggregate::create_weights(FFModel& model)
{
  // Do nothing
}


void Aggregate::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
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


  // Compute partition bound for input TODO: ???????
  for(int i = 0; i < n+2; i++) {
    Rect<2> input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[i].part.get_index_partition());
    if (input_rect == part_rect) {
      input_lps[i] = inputs[i].part;
      input_grad_lps[i] = inputs[i].part_grad;
    } else {
      model.create_disjoint_partition<2>(
        inputs[i], (IndexSpaceT<2>)task_is, input_lps[i], input_grad_lps[i]);
    }
  }
}



// TODO: ?
OpMeta* Aggregate::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  //FFHandler handle = *((FFHandler*)task->local_args);
  //TopKMeta* m = new TopKMeta(handle);
  //return m;
  // return NULL if we don't need local metadata for Aggregate
  return NULL;
}


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
      READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/, //TODO ?
      READ_WRITE, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);
  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i+2], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+2].region));
    launcher.add_field(i+2, FID_DATA);
  }
  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, outputs[0].region));
  launcher.add_field(n+2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}


void aggregate_forward(float** exp_preds,
        const int* exp_assign,
        const float* gating_net_preds,
        float* output,
        int n, // num experts
        int k, // num chosen experts
        int exp_samples,
        int batch_size,
        int out_dim)
{
  std::vector<int> expert_idx(n, 0);
  float* chosen_exp_preds[k];
  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < k; j++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[k*i + j];
      if(expert >= exp_samples) {
        chosen_exp_preds[j] = 0;
        continue;
      }
      chosen_exp_preds[j] = exp_preds[expert] + expert_idx[expert]*out_dim;
      expert_idx[expert]++;
    }
    f_first(output+i*out_dim, gating_net_preds+i*out_dim, chosen_exp_preds,
      k, out_dim);
  }
}


void aggregate_backward(float** exp_preds,
        const int* exp_assign,
        const float* gating_net_preds,
        float* output,
        int n, // num experts TODO: int
        int k, // num chosen experts
        int batch_size,
        int out_dim)
{

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

  // get exp_preds
  float* exp_preds[n];
  // get first exp_pred and row and out_dim
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerWO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  coord_t rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
  coord_t out_dim = out_domain.hi()[0] - out_domain.lo()[0] + 1;

  for(int i = 1; i < n; i++) {
    out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerWO<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    assert(rows == out_domain.hi()[1] - out_domain.lo()[1] + 1);
    assert(out_dim == out_domain.hi()[0] - out_domain.lo()[0] + 1);
  }

  coord_t k = rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1;

  aggregate_forward(exp_preds, acc_gate_assign.ptr(rect_gate_assign),
    acc_gate_pred.ptr(rect_gate_pred), acc_output.ptr(rect_output), n, k, rows,
    batch_size, out_dim);
}

void Aggregate::backward_task(const Task *task,
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

  // get exp_preds
  float* exp_preds[n];
  // get first exp_pred and row and out_dim
  Domain out_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerWO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  coord_t rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
  coord_t out_dim = out_domain.hi()[0] - out_domain.lo()[0] + 1;

  for(int i = 1; i < n; i++) {
    out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerWO<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    assert(rows == out_domain.hi()[1] - out_domain.lo()[1] + 1);
    assert(out_dim == out_domain.hi()[0] - out_domain.lo()[0] + 1);
  }

  coord_t k = rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1;

  aggregate_backward(exp_preds, acc_gate_assign.ptr(rect_gate_assign),
    acc_gate_pred.ptr(rect_gate_pred), acc_output.ptr(rect_output), n, k,
    batch_size, out_dim);
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
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/, //TODO ?
      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);
  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i+2], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i+2].region));
    launcher.add_field(i+2, FID_DATA);
  }
  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
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
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);

  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[1], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1].region_grad));
  launcher.add_field(1, FID_DATA);

  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[i+2], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+2].region_grad));
    launcher.add_field(i+2, FID_DATA);
  }

  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(n+2, FID_DATA);

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

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
#include <math.h>

using namespace Legion;

void FFModel::group_by(const Tensor input,
                       const Tensor assign,
                       Tensor* outputs,
                       int n, float alpha,
                       const char* name)
{
  Group_by* group_by = new Group_by(*this, input, assign, n, alpha, name);
  layers.push_back(group_by);
  for (int i = 0; i < n; i++)
    outputs[i] = group_by->outputs[i];
}


Group_by::Group_by(FFModel& model,
                  const Tensor _input,
                  const Tensor _assign,
                  int _n, float _alpha,
                  const char* name)
: Op(model, OP_GROUP_BY, name, 2/*inputs*/, 0/*weights*/, _n/*outputs*/, _input, _assign),
  n(_n),
  alpha(_alpha)
  //profiling(model.config.profiling)
{
  assert(_input->num_dims == 2); // NOTE: Is that a problem if you e.g. want to pass in images
  assert(_input->num_dims == 2);
  assert(_input->dims[1] == _assign->dims[1]);
  assert(n > 0);

  // List of outputs
  int k = _assign->dims[0].size;
  for(int i = 0; i < n; i++) {
    outputs[i]->num_dims = 2;
    outputs[i]->dims[0].size = inputs[0]->dims[0].size;
    outputs[i]->dims[1].size = (int)ceil(alpha*k/n*inputs[0]->dims[1].size);
  }

  numWeights = 0;
}

OpMeta* Group_by::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  return NULL;
}

void Group_by::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(GROUP_BY_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // data
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // output
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region));
    launcher.add_field(i+2, FID_DATA);
  }

  runtime->execute_index_space(ctx, launcher);
}


void group_by_forward(const float* input,
        const int* exp_assign,
        float** outputs,
        int n, // num experts
        int k, // chosen experts
        float alpha, // factor additional memory assigned
        int batch_size,
        int out_dim)
{
  std::vector<int> expert_idx(n, 0);
  int exp_tensor_rows = ceil(alpha*k/n*batch_size);

  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < k; j++) {
      int expert = exp_assign[i*k + j];
      int row = expert_idx[expert];

      if(row >= exp_tensor_rows)
        continue; // dropped sample

      // copy over sample
      int input_start = i*out_dim;
      for(int l = 0; l < out_dim; l++) {
        outputs[expert][row*out_dim + l] = input[input_start + l];
      }
      expert_idx[expert]++;
    }
  }
}


void group_by_backward(float* input_grad,
        const int* exp_assign,
        float** output_grads,
        int n, // num experts
        int k, // chosen experts
        float alpha, // factor additional memory assigned
        int batch_size,
        int out_dim)
{
  std::vector<int> expert_idx(n, 0);
  int exp_tensor_rows = ceil(alpha*k/n*batch_size);

  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < k; j++) {
      int expert = exp_assign[i*k + j];
      int row = expert_idx[expert];

      if(row >= exp_tensor_rows)
        continue; // dropped sample

      // add gradient
      int input_start = i*out_dim;
      for(int l = 0; l < out_dim; l++) {
        input_grad[input_start + l] += output_grads[expert][row*out_dim + l]/k;
      }
      expert_idx[expert]++;
    }
  }
}


void Group_by::forward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  // Get n, alpha
  const Group_by* gb = (Group_by*) task->args;
  int n = gb->n;
  float alpha = gb->alpha;

  assert((int)regions.size() == n+2);
  assert((int)task->regions.size() == n+2);

  // get input and assign regions
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_assign(regions[1], FID_DATA);

  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t input_cols = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // get output
  float* outputs[n];
  //int exp_output_rows = (int)ceil(alpha*k/n*batch_size);
  for(int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    outputs[i] = helperGetTensorPointerWO<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    //coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    //assert((int)output_rows == exp_output_rows);
    assert(output_cols == input_cols);
  }

  group_by_forward(acc_input.ptr(rect_input), acc_assign.ptr(rect_assign),
      outputs, n, k, alpha, batch_size, data_dim);
}


void Group_by::backward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  // Get n, alpha
  const Group_by* gb = (Group_by*) task->args;
  int n = gb->n;
  float alpha = gb->alpha;

  assert((int)regions.size() == n+2);
  assert((int)task->regions.size() == n+2);

  // get input and assign regions
  const AccessorWO<float, 2> acc_input_grad(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_assign(regions[1], FID_DATA);

  Rect<2> rect_input_grad = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input_grad.hi[1] - rect_input_grad.lo[1] + 1;
  coord_t input_cols = rect_input_grad.hi[0] - rect_input_grad.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // get output
  float* output_grads[n];
  //int exp_output_rows = (int)ceil(alpha*k/n*batch_size);
  for(int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    output_grads[i] = helperGetTensorPointerRW<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    //coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    //assert((int)output_rows == exp_output_rows);
    assert(output_cols == input_cols);
  }

  group_by_backward(acc_input_grad.ptr(rect_input_grad), acc_assign.ptr(rect_assign),
      output_grads, n, k, alpha, batch_size, data_dim);
}


void Group_by::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(GROUP_BY_FWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // data
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);

  // assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // output
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region));
    launcher.add_field(i+2, FID_DATA);
  }

  runtime->execute_index_space(ctx, launcher);
}

void Group_by::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(GROUP_BY_BWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());

  // input_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);

  // assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // output grad
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part_grad, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region_grad));
    launcher.add_field(i+2, FID_DATA);
  }

  runtime->execute_index_space(ctx, launcher);
}


bool Group_by::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics) const
{
  //TODO: implement
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.memory_requirement = 0;
  return false;
}

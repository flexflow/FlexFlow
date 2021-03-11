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


Tensor FFModel::group_by(const Tensor& input,
                        const Tensor& assign,
                        int n, int k, float alpha,
                        const char* name)
{
  Group_by* group_by = new Group_by(*this, input, assign, n, k, alpha, name);
  layers.push_back(group_by);
  return group_by->outputs[0];
}


Group_by::Group_by(FFModel& model,
                  const Tensor& _input,
                  const Tensor& _assign,
                  int _n, int _k, float _alpha,
                  const char* name)
: Op(model, OP_GROUP_BY, name, _input, _assign),
  n(_n),
  k(_k),
  alpha(_alpha)
  //profiling(model.config.profiling)
{
  assert(_input.numDim == 2); // NOTE: Is that a problem if you e.g. want to pass in images
  assert(_input.numDim == 2);
  assert(_input.adim[1] == _assign.adim[1]);
  assert(_assign.adim[0] == k);
  assert(n >= k);

  // TODO: Could have list as output, see split
  outputs[0].numDim = 2;
  outputs[0].adim[0] = inputs[0].adim[0];
  outputs[0].adim[1] = (int)(alpha*k*inputs[0].adim[1]);

  numWeights = 0;
}


void Group_by::create_weights(FFModel& model)
{
  // Do nothing
}


void Group_by::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);

  // Currently assume we can only partition over the sample dim
  assert(part_rect.hi[0] == part_rect.lo[0]);

  const int dims[2] = {(int)(alpha*k*inputs[0].adim[1]), inputs[0].adim[0]};
  outputs[0] = model.create_tensor<2>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;

  // Compute partition bound for input
  Rect<2> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_lps[1] = inputs[1].part;
    input_grad_lps[0] = inputs[0].part_grad;
    input_grad_lps[1] = inputs[1].part_grad;
  } else {
    model.create_disjoint_partition<2>(
      inputs[0], (IndexSpaceT<2>)task_is, input_lps[0], input_grad_lps[0]);
    model.create_disjoint_partition<2>(
      inputs[1], (IndexSpaceT<2>)task_is, input_lps[1], input_grad_lps[1]);
  }
}


// TODO: ?
OpMeta* Group_by::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  FFHandler handle = *((FFHandler*)task->local_args);
  TopKMeta* m = new TopKMeta(handle);
  return m;
}


void Group_by::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(GROUP_BY_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // output TODO: List
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);

  // data
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(1, FID_DATA);

  // assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1].part, 0/*projection id*/, //TODO ?
      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(2, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}


void group_by_forward(const float* input,
        const int* exp_assign,
        float* output,
        int n, // num experts
        int k, // chosen experts
        float alpha, // factor additional memory assigned
        int batch_size,
        int out_dim)
{
  std::vector<int> expert_idx(n, 0);
  int exp_tensor_rows = alpha*k/n*batch_size;
  int exp_tensor_entries = exp_tensor_rows * out_dim;

  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < k; j++) {
      int expert = exp_assign[i*k + j];
      int row = expert_idx[expert];

      if(row >= exp_tensor_rows)
        continue;

      // copy over sample (maybe better memcpy or so)
      int output_start = expert*exp_tensor_entries + row*out_dim;
      int input_start = i*out_dim;
      for(int l = 0; l < out_dim; l++) {
        output[output_start + l] = input[input_start + l];
      }
      expert_idx[expert]++;
    }
  }
}


void group_by_backward(const float* input,
        const int* exp_assign,
        float* output,
        int n, // num experts
        int k, // chosen experts
        float alpha, // factor additional memory assigned*/
        int batch_size,
        int out_dim)
{
  // TODO: Implement. In case not data passed directly (this is uncommon though)
}


void Group_by::forward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  // Get n, k, alpha
  const Group_by* gb = (Group_by*) task->args;
  int n = gb->n;
  int k = gb->k;
  float alpha = gb->alpha;

  // get regions
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA); // TODO: We can make this 3D, more elegant
  const AccessorRO<int, 2> acc_assign(regions[2], FID_DATA);

  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  // Get data shapes and rnsure they match
  coord_t input_rows = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t output_rows = rect_output.hi[1] - rect_output.lo[1] + 1;
  int batch_size = input_rows;
  assert((int)(alpha*k*(int)output_rows) == batch_size);

  coord_t input_cols = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(input_cols == rect_output.hi[0] - rect_output.lo[0] + 1);
  int data_dim = input_cols;

  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  assert(k == (int)(rect_assign.hi[0] - rect_assign.lo[0] + 1));

  group_by_forward(acc_input.ptr(rect_input), acc_assign.ptr(rect_assign),
      acc_output.ptr(rect_output), n, k, alpha, batch_size, data_dim);
}


void Group_by::backward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  // Get n, k, alpha
  const Group_by* gb = (Group_by*) task->args;
  int n = gb->n;
  int k = gb->k;
  float alpha = gb->alpha;

  // get regions
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<int, 2> acc_assign(regions[2], FID_DATA);

  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  // Get data shapes and ensure they match
  coord_t input_rows = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t output_rows = rect_output.hi[1] - rect_output.lo[1] + 1;
  int batch_size = input_rows;
  assert((int)(alpha*k*(int)output_rows) == batch_size);

  coord_t input_cols = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(input_cols == rect_output.hi[0] - rect_output.lo[0] + 1);
  int data_dim = input_cols;

  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  assert(k == (int)(rect_assign.hi[0] - rect_assign.lo[0] + 1));

  group_by_backward(acc_input.ptr(rect_input), acc_assign.ptr(rect_assign),
      acc_output.ptr(rect_output), n, k, alpha, batch_size, data_dim);
}


bool Group_by::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics)
{
  // TODO: To be implemented
  assert(false);
  return false;
}


void Group_by::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(GROUP_BY_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // output TODO: List
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);

  // data
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(1, FID_DATA);

  // assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1].part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(2, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

void Group_by::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(GROUP_BY_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // output TODO: List
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(0, FID_DATA);

  // data
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(1, FID_DATA);

  // assign -- not gradients
  launcher.add_region_requirement(
    RegionRequirement(inputs[1].part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(2, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

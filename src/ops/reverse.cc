/* Copyright 2020 Facebook
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

#include "flexflow/ops/reverse.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;

Tensor FFModel::reverse(const Tensor input,
                        int axis,
                        const char* name)
{
  Reverse* reverse = new Reverse(*this, input, axis, name);
  layers.push_back(reverse);
  return reverse->outputs[0];
}

Reverse::Reverse(FFModel& model,
                 const Tensor input,
                 int _axis,
                 const char* name)
: Op(model, OP_REVERSE, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, input), axis(_axis)
{
  numOutputs = 1;
  int numdim = input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++)
    dims[i] = input->dims[i];
  outputs[0] = model.create_tensor_legion_ordering(numdim, dims, input->data_type, this);
}

#ifdef DEADCODE
void Reverse::create_input_partition(FFModel& model)
{
  // Retrive the task indexspace
  int dim = inputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      task_is = model.get_or_create_task_is(DIM, name); \
      create_input_partition_with_dim<DIM>(model); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      // Unsupported dim for Reverse operator
      assert(false);
    }
  }
}

template<int NDIM>
void Reverse::create_input_partition_with_dim(FFModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // the degree of parallelism along the reversed dimension must be 1
  assert(part_rect.hi[NDIM-1-axis] == part_rect.lo[NDIM-1-axis]);
  return Op::create_input_partition(model);
#ifdef DEADCODE
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = outputs[0].dims[NDIM-1-i].size;
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  Rect<NDIM> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0]->part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0]->part;
    input_grad_lps[0] = inputs[0]->part_grad;
  } else {
    model.create_disjoint_partition<NDIM>(
        inputs[0], IndexSpaceT<NDIM>(task_is), input_lps[0], input_grad_lps[0]);
  }
#endif
}
#endif

void Reverse::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(REVERSE_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Reverse)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Reverse::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(REVERSE_FWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Reverse)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Reverse::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(REVERSE_BWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Reverse)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input0_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

}; // namespace FlexFlow
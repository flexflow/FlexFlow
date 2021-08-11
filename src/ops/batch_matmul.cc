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

#include "flexflow/ops/batch_matmul.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using PCG::Node;

Tensor FFModel::batch_matmul(const Tensor A,
                             const Tensor B,
                             int a_seq_length_dim,
                             int b_seq_length_dim)
{
  BatchMatmul* bmm = new BatchMatmul(*this, A, B,
      a_seq_length_dim, b_seq_length_dim);
  layers.push_back(bmm);
  return bmm->outputs[0];
}

// return A*B
BatchMatmul::BatchMatmul(FFModel& model,
                         const Tensor A,
                         const Tensor B,
                         int _a_seq_length_dim,
                         int _b_seq_length_dim)
: Op(model, OP_BATCHMATMUL, "BatchMatmul_", 2/*inputs*/, 0/*weights*/, 1/*outputs*/, A, B),
  a_seq_length_dim(A->num_dims-1-_a_seq_length_dim),
  b_seq_length_dim(B->num_dims-1-_b_seq_length_dim)
{
  assert((a_seq_length_dim <= 1) && "FlexFlow currently only supports seq_length_dim of 0 or 1 (in Fortran ordering).");
  assert((b_seq_length_dim <= 1) && "FlexFlow currently only supports seq_length_dim of 0 or 1 (in Fortran ordering).");
  assert(A->num_dims == B->num_dims);
  for (int i = A->num_dims-1; i >= 2; i--)
    assert(A->dims[i] == B->dims[i]);
  assert(A->dims[0] == B->dims[1]);
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < A->num_dims; i++)
    dims[i] = A->dims[i];
  dims[0] = B->dims[0];
  numOutputs = 1;
  outputs[0] = model.create_tensor_legion_ordering(A->num_dims, dims, DT_FLOAT, this);
  // C is not none
  //if (C != Tensor::NO_TENSOR) {
  //  numInputs = 3;
  //  assert(C.num_dims == outputs[0].num_dims);
  //  for (int i = 0; i < C.num_dims; i++)
  //    assert(C.adim[i] == outputs[0].adim[i]);
  //}
}

#ifdef DEADCODE
void BatchMatmul::create_input_partition(FFModel& model)
{
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Domain part_rect = runtime->get_index_space_domain(ctx, task_is);
  // currently only support data parallel for batch matmul
  // the parallel degree of the inner most two dims must be 1
  assert(part_rect.hi()[0] == part_rect.lo()[0]);
  assert(part_rect.hi()[1] == part_rect.lo()[1]);
  return Op::create_input_partition(model);
#ifdef DEADCODE
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = outputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  for (int i = 0; i < numInputs; i++) {
    Rect<NDIM> input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[i]->part.get_index_partition());
    if (input_rect == part_rect) {
      input_lps[i] = inputs[i]->part;
      input_grad_lps[i] = inputs[i]->part_grad;
    } else {
      model.create_disjoint_partition<NDIM>(
          inputs[i], IndexSpaceT<NDIM>(task_is), input_lps[i], input_grad_lps[i]);
    }
  }
#endif
}
#endif

void BatchMatmul::init(const FFModel& ff)
{
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      init_with_dim<DIM>(ff); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
void BatchMatmul::init_with_dim(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(BATCHMATMUL_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(BatchMatmul)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i]->region));
    launcher.add_field(i+1, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void BatchMatmul::forward(const FFModel& ff)
{
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      forward_with_dim<DIM>(ff); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template<int NDIM>
void BatchMatmul::forward_with_dim(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(BATCHMATMUL_FWD_TASK_ID, parallel_is,
      TaskArgument(&ff.iter_config, sizeof(FFIterationConfig)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i]->region));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void BatchMatmul::backward(const FFModel& ff)
{
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      backward_with_dim<DIM>(ff); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

/*
  regions[0](I): output
  regions[1](I): output_grad
  regions[2](I): A
  regions[3](I/O): A_grad
  regions[4](I): B
  regions[5](I/O): B_grad
  regions[6](I/O): C_grad
*/
template<int NDIM>
void BatchMatmul::backward_with_dim(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(BATCHMATMUL_BWD_TASK_ID, parallel_is,
      TaskArgument(&ff.iter_config, sizeof(FFIterationConfig)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  // regions[0](I): output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): A
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I/O): A_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(3, FID_DATA);
  // regions[4](I): B
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(4, FID_DATA);
  // regions[5](I/O): B_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1]->region_grad));
  launcher.add_field(5, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void BatchMatmul::print_layer(const FFModel& ff)
{
  return;
}

}; // namespace FlexFlow
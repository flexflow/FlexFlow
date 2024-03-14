/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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
#include "flexflow/ops/kernels/batch_matmul_kernels.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;
using PCG::Node;

using namespace FlexFlow::Kernels::BatchMatmul;

bool operator==(BatchMatmulParams const &lhs, BatchMatmulParams const &rhs) {
  return lhs.a_seq_length_dim == rhs.a_seq_length_dim &&
         lhs.a_seq_length_dim == rhs.a_seq_length_dim;
}

bool BatchMatmulParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const {
  if (!input.first.is_valid()) {
    return false;
  }
  if (!input.second.is_valid()) {
    return false;
  }
  if (input.first.num_dims != input.second.num_dims) {
    return false;
  }
  for (int i = input.first.num_dims - 1; i >= 2; i--) {
    if (input.first.dims[i] != input.second.dims[i]) {
      return false;
    }
  }
  if (input.first.dims[0] != input.second.dims[1]) {
    return false;
  }
  return true;
}

BatchMatmulParams BatchMatmul::get_params() const {
  BatchMatmulParams params;
  params.a_seq_length_dim = inputs[0]->num_dims - 1 - this->a_seq_length_dim;
  params.b_seq_length_dim = inputs[1]->num_dims - 1 - this->b_seq_length_dim;
  return params;
}

Tensor FFModel::batch_matmul(const Tensor A,
                             const Tensor B,
                             int a_seq_length_dim,
                             int b_seq_length_dim,
                             char const *name) {
  Layer *bmm = new Layer(this,
                         OP_BATCHMATMUL,
                         DT_FLOAT,
                         name,
                         2 /*inputs*/,
                         0 /*weights*/,
                         1 /*outputs*/,
                         A,
                         B);
  assert((a_seq_length_dim <= 1) &&
         "FlexFlow currently only supports seq_length_dim of 0 or 1 (in "
         "Fortran ordering).");
  assert((b_seq_length_dim <= 1) &&
         "FlexFlow currently only supports seq_length_dim of 0 or 1 (in "
         "Fortran ordering).");
  assert(A->num_dims == B->num_dims);
  for (int i = A->num_dims - 1; i >= 2; i--) {
    assert(A->dims[i] == B->dims[i]);
  }
  assert(A->dims[0] == B->dims[1]);
  int dims[MAX_TENSOR_DIM];
  int numdim = A->num_dims;
  for (int i = 0; i < numdim; i++) {
    dims[i] = A->dims[i];
  }
  dims[0] = B->dims[0];
  bmm->outputs[0] = create_tensor_legion_ordering(
      numdim, dims, A->data_type, bmm, 0, true /*create_grad*/);
  bmm->add_int_property("a_seq_length_dim", a_seq_length_dim);
  bmm->add_int_property("b_seq_length_dim", b_seq_length_dim);
  layers.push_back(bmm);
  return bmm->outputs[0];
}

Op *BatchMatmul::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("a_seq_length_dim", value);
  int a_seq_length_dim = value;
  layer->get_int_property("b_seq_length_dim", value);
  int b_seq_length_dim = value;
  return new BatchMatmul(model,
                         inputs[0],
                         inputs[1],
                         a_seq_length_dim,
                         b_seq_length_dim,
                         layer->name);
}

BatchMatmul::BatchMatmul(
    FFModel &model,
    BatchMatmulParams const &params,
    std::pair<ParallelTensor, ParallelTensor> const &inputs,
    char const *name)
    : BatchMatmul(model,
                  inputs.first,
                  inputs.second,
                  params.a_seq_length_dim,
                  params.b_seq_length_dim,
                  params.name) {}

// return A*B
BatchMatmul::BatchMatmul(FFModel &model,
                         const ParallelTensor A,
                         const ParallelTensor B,
                         int _a_seq_length_dim,
                         int _b_seq_length_dim,
                         char const *name)
    : Op(model,
         OP_BATCHMATMUL,
         DT_FLOAT,
         name,
         2 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         A,
         B),
      a_seq_length_dim(A->num_dims - 1 - _a_seq_length_dim),
      b_seq_length_dim(B->num_dims - 1 - _b_seq_length_dim) {
  assert((_a_seq_length_dim <= 1) &&
         "FlexFlow currently only supports seq_length_dim of 0 or 1 (in "
         "Fortran ordering).");
  assert((_b_seq_length_dim <= 1) &&
         "FlexFlow currently only supports seq_length_dim of 0 or 1 (in "
         "Fortran ordering).");
  assert(A->num_dims == B->num_dims);
  for (int i = A->num_dims - 1; i >= 2; i--) {
    assert(A->dims[i] == B->dims[i]);
  }
  assert(A->dims[0] == B->dims[1]);
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < A->num_dims; i++) {
    dims[i] = A->dims[i];
  }
  dims[0] = B->dims[0];
  numOutputs = 1;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      A->num_dims, dims, DT_FLOAT, this);
  // C is not none
  // if (C != Tensor::NO_TENSOR) {
  //  numInputs = 3;
  //  assert(C.num_dims == outputs[0].num_dims);
  //  for (int i = 0; i < C.num_dims; i++)
  //    assert(C.adim[i] == outputs[0].adim[i]);
  //}
}

void BatchMatmul::serialize(Legion::Serializer &sez) const {
  BatchMatmulParams params = get_params();
  sez.serialize(params.a_seq_length_dim);
  sez.serialize(params.b_seq_length_dim);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;
/*static*/
Node BatchMatmul::deserialize(FFModel &ff,
                              Legion::Deserializer &dez,
                              ParallelTensor inputs[],
                              int num_inputs) {
  assert(num_inputs == 2);
  int a_seq_length_dim, b_seq_length_dim;
  dez.deserialize(a_seq_length_dim);
  dez.deserialize(b_seq_length_dim);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);

  BatchMatmulParams params;
  params.a_seq_length_dim = a_seq_length_dim;
  params.b_seq_length_dim = b_seq_length_dim;
  strcpy(params.name, name);
  return ff.get_or_create_node<BatchMatmul>({inputs[0], inputs[1]}, params);
}

Op *BatchMatmul::materialize(FFModel &ff,
                             ParallelTensor inputs[],
                             int num_inputs) const {
  BatchMatmulParams params = get_params();
  return new BatchMatmul(ff, params, {inputs[0], inputs[1]}, this->name);
}

void BatchMatmul::init(FFModel const &ff) {
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    init_with_dim<DIM>(ff);                                                    \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template <int NDIM>
void BatchMatmul::init_with_dim(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(BATCHMATMUL_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(BatchMatmul)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(RegionRequirement(inputs[i]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[i]->region));
    launcher.add_field(i + 1, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *BatchMatmul::init_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  BatchMatmul const *bmm = (BatchMatmul *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  BatchMatmulMeta *m = new BatchMatmulMeta(handle);
  m->profiling = bmm->profiling;
  m->inference_debugging = bmm->inference_debugging;
  m->a_seq_length_dim = bmm->a_seq_length_dim;
  m->b_seq_length_dim = bmm->b_seq_length_dim;
  std::strcpy(m->op_name, bmm->name);
  m->layer_guid = bmm->layer_guid;
  return m;
}

void BatchMatmul::forward(FFModel const &ff) {
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    forward_with_dim<DIM>(ff);                                                 \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

template <int NDIM>
void BatchMatmul::forward_with_dim(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(
      BATCHMATMUL_FWD_TASK_ID,
      parallel_is,
      TaskArgument(&ff.iter_config, sizeof(FFIterationConfig)),
      argmap,
      Predicate::TRUE_PRED,
      false /*must*/,
      0 /*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(RegionRequirement(inputs[i]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[i]->region));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](O): output
  regions[1](I): A
  regions[2](I): B
  (optional) regions[3](I): C
  output = A * B + C
*/
void BatchMatmul::forward_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // const BatchMatmul* bmm = (const BatchMatmul*) task->args;
  FFIterationConfig const *iter_config = (FFIterationConfig const *)task->args;
  BatchMatmulMeta const *meta = *((BatchMatmulMeta **)task->local_args);
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain a_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain b_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  int m = b_domain.hi()[0] - b_domain.lo()[0] + 1;
  assert(m == out_domain.hi()[0] - out_domain.lo()[0] + 1);
  int n = a_domain.hi()[1] - a_domain.lo()[1] + 1;
  assert(n == out_domain.hi()[1] - out_domain.lo()[1] + 1);
  int k = a_domain.hi()[0] - a_domain.lo()[0] + 1;
  assert(k == b_domain.hi()[1] - b_domain.lo()[1] + 1);
  assert(a_domain.get_dim() == b_domain.get_dim());
  assert(a_domain.get_dim() == out_domain.get_dim());
  int batch = 1;
  for (int i = 2; i < a_domain.get_dim(); i++) {
    int dim_size = a_domain.hi()[i] - a_domain.lo()[i] + 1;
    assert(dim_size == b_domain.hi()[i] - b_domain.lo()[i] + 1);
    assert(dim_size == out_domain.hi()[i] - out_domain.lo()[i] + 1);
    batch *= dim_size;
  }
  float *out_ptr = helperGetTensorPointerWO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float const *a_ptr = helperGetTensorPointerRO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float const *b_ptr = helperGetTensorPointerRO<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  float const *c_ptr = NULL;
  if (regions.size() == 4) {
    Domain c_domain = runtime->get_index_space_domain(
        ctx, task->regions[3].region.get_index_space());
    assert(c_domain == a_domain);
    c_ptr = helperGetTensorPointerRO<float>(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
  }

  forward_kernel_wrapper(meta,
                         out_ptr,
                         a_ptr,
                         b_ptr,
                         c_ptr,
                         m,
                         n,
                         k,
                         batch,
                         meta->a_seq_length_dim,
                         meta->b_seq_length_dim,
                         iter_config->seq_length);
}

void BatchMatmul::backward(FFModel const &ff) {
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    backward_with_dim<DIM>(ff);                                                \
    break;                                                                     \
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
template <int NDIM>
void BatchMatmul::backward_with_dim(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(
      BATCHMATMUL_BWD_TASK_ID,
      parallel_is,
      TaskArgument(&ff.iter_config, sizeof(FFIterationConfig)),
      argmap,
      Predicate::TRUE_PRED,
      false /*must*/,
      0 /*mapper_id*/,
      outputs[0]->machine_view.hash());
  // regions[0](I): output
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): A
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I/O): A_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(3, FID_DATA);
  // regions[4](I): B
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(4, FID_DATA);
  // regions[5](I/O): B_grad
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[1]->region_grad));
  launcher.add_field(5, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
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
__host__ void
    BatchMatmul::backward_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  // Currently assume C is NULL
  assert(regions.size() == 6);
  assert(task->regions.size() == 6);
  // BatchMatmul* bmm = (BatchMatmul*) task->args;
  FFIterationConfig const *iter_config = (FFIterationConfig const *)task->args;
  BatchMatmulMeta const *meta = *((BatchMatmulMeta **)task->local_args);
  // output domains
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(out_domain == out_grad_domain);
  // A domains
  Domain a_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  Domain a_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
  assert(a_domain == a_grad_domain);
  // B domains
  Domain b_domain = runtime->get_index_space_domain(
      ctx, task->regions[4].region.get_index_space());
  Domain b_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[4].region.get_index_space());
  assert(b_domain == b_grad_domain);
  // check dins
  int m = b_domain.hi()[0] - b_domain.lo()[0] + 1;
  assert(m == out_domain.hi()[0] - out_domain.lo()[0] + 1);
  int n = a_domain.hi()[1] - a_domain.lo()[1] + 1;
  assert(n == out_domain.hi()[1] - out_domain.lo()[1] + 1);
  int k = a_domain.hi()[0] - a_domain.lo()[0] + 1;
  assert(k == b_domain.hi()[1] - b_domain.lo()[1] + 1);
  assert(a_domain.get_dim() == b_domain.get_dim());
  assert(a_domain.get_dim() == out_domain.get_dim());
  int batch = 1;
  for (int i = 2; i < a_domain.get_dim(); i++) {
    int dim_size = a_domain.hi()[i] - a_domain.lo()[i] + 1;
    assert(dim_size == b_domain.hi()[i] - b_domain.lo()[i] + 1);
    assert(dim_size == out_domain.hi()[i] - out_domain.lo()[i] + 1);
    batch *= dim_size;
  }
  // get pointers
  float const *out_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float const *out_grad_ptr = helperGetTensorPointerRO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float const *a_ptr = helperGetTensorPointerRO<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  float *a_grad_ptr = helperGetTensorPointerRW<float>(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  float const *b_ptr = helperGetTensorPointerRO<float>(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  float *b_grad_ptr = helperGetTensorPointerRW<float>(
      regions[5], task->regions[5], FID_DATA, ctx, runtime);

  float *c_grad_ptr = NULL;

  // TODO: add support for meta->a_seq_length_dim >= 0
  // or meta->b_seq_length_dim >= 0
  assert((meta->a_seq_length_dim >= a_domain.get_dim()) ||
         (iter_config->seq_length == 0));
  assert((meta->b_seq_length_dim >= b_domain.get_dim()) ||
         (iter_config->seq_length == 0));

  backward_kernel_wrapper(meta,
                          out_ptr,
                          out_grad_ptr,
                          a_ptr,
                          a_grad_ptr,
                          b_ptr,
                          b_grad_ptr,
                          c_grad_ptr,
                          m,
                          n,
                          k,
                          batch);
}

void BatchMatmul::print_layer(FFModel const &ff) {
  return;
}

bool BatchMatmul::measure_operator_cost(Simulator *sim,
                                        MachineView const &pc,
                                        CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input0, sub_input1;
  if (!outputs[0]->get_sub_tensor(pc, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(pc, sub_input0)) {
    return false;
  }
  if (!inputs[1]->get_sub_tensor(pc, sub_input1)) {
    return false;
  }

  int input0_c = sub_input0.dims[0].size;
  int input0_r = sub_input0.dims[1].size;
  int input1_c = sub_input1.dims[0].size;
  int input1_r = sub_input1.dims[1].size;
  int output_c = sub_output.dims[0].size;
  int output_r = sub_output.dims[1].size;

  assert(input0_c == input1_r);
  assert(input0_r == output_r);
  assert(input1_c == output_c);

  assert(sub_input0.dims[2] == sub_input1.dims[2]);
  assert(sub_input1.dims[2] == sub_output.dims[2]);
  int batch = 1;
  assert(sub_input0.num_dims == sub_input1.num_dims);
  for (int i = 2; i < sub_input0.num_dims; i++) {
    assert(sub_input0.dims[i] == sub_input1.dims[i]);
    assert(sub_input0.dims[i] == sub_output.dims[i]);
    batch *= sub_input0.dims[i].size;
  }

  BatchMatmulMeta *meta = sim->batch_matmul_meta;

  // allocate tensors in simulator
  sim->free_all();
  float *a_ptr = (float *)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
  assert(a_ptr != NULL);
  float *b_ptr = (float *)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
  assert(b_ptr != NULL);
  float *c_ptr = NULL;
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *out_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(out_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  int m = input1_c;
  int n = input0_r;
  int k = input0_c;

  assert(meta->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(meta, out_ptr, a_ptr, b_ptr, c_ptr, m, n, k, batch);
  };

  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *a_grad_ptr =
        (float *)sim->allocate(sub_input0.get_volume(), DT_FLOAT);
    float *b_grad_ptr =
        (float *)sim->allocate(sub_input1.get_volume(), DT_FLOAT);
    float *c_grad_ptr = NULL;
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *out_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(out_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [&] {
      backward_kernel_wrapper(meta,
                              out_ptr,
                              out_grad_ptr,
                              a_ptr,
                              a_grad_ptr,
                              b_ptr,
                              b_grad_ptr,
                              c_grad_ptr,
                              m,
                              n,
                              k,
                              batch);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure BatchMatmul] name(%s) adim(%d %d %d) bdim(%d %d %d) "
           "odim(%d %d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
           name,
           batch,
           input0_r,
           input0_c,
           batch,
           input1_r,
           input1_c,
           batch,
           output_r,
           output_c,
           cost_metrics.forward_time,
           cost_metrics.backward_time);
  } else {
    printf("[Measure BatchMatmul] name(%s) adim(%d %d %d) bdim(%d %d %d) "
           "odim(%d %d %d) forward_time(%.4lf)\n",
           name,
           batch,
           input0_r,
           input0_c,
           batch,
           input1_r,
           input1_c,
           batch,
           output_r,
           output_c,
           cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::BatchMatmulParams>::operator()(
    FlexFlow::BatchMatmulParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.a_seq_length_dim);
  hash_combine(key, params.b_seq_length_dim);
  return key;
}
}; // namespace std

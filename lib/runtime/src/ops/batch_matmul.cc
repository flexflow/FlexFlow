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

#include "batch_matmul.h"
#include "kernels/batch_matmul_kernels.h"
#include "kernels/profiling.h"
#include "legion/legion_utilities.h"
#include "tasks.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::BatchMatmul;

enum Slots {
  A_INPUT,
  B_INPUT,
  OUTPUT,
  A_INPUT_GRAD,
  B_INPUT_GRAD,
  OUTPUT_GRAD,
  ATTRS,
  PROFILING
};

OpTaskInvocation init(BatchMatmulAttrs const &attrs) {
  OpTaskBinding b;

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, enable_profiling());

  return {BATCHMATMUL_INIT_TASK_ID, b};
}

OpTaskInvocation forward(BatchMatmulAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(A_INPUT, input_tensor(0));
  b.bind(B_INPUT, input_tensor(1));
  b.bind(OUTPUT, output_tensor(0));

  return {BATCHMATMUL_FWD_TASK_ID, b};
}

OpTaskInvocation backward(BatchMatmulAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {BATCHMATMUL_BWD_TASK_ID, b};
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
                  name) {}

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

  BatchMatmulParams params;
  params.a_seq_length_dim = a_seq_length_dim;
  params.b_seq_length_dim = b_seq_length_dim;
  return ff.get_or_create_node<BatchMatmul>({inputs[0], inputs[1]}, params);
}

Op *BatchMatmul::materialize(FFModel &ff,
                             ParallelTensor inputs[],
                             int num_inputs) const {
  BatchMatmulParams params = get_params();
  return new BatchMatmul(ff, params, {inputs[0], inputs[1]}, this->name);
}

template <>
void register_task<BATCHMATMUL_INIT_TASK_ID>() {
  OpTaskSignature sig(OpTaskType::INIT);

  sig.add_arg_slot<BatchMatmulAttrs>(ATTRS);
  sig.add_arg_slot<bool>(PROFILING);

  register_task(BATCHMATMUL_INIT_TASK_ID, "BatchMatmul Init", sig, init_task);
}

static OpTaskSignature get_fwd_task_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(A_INPUT, READ_WRITE);
  fwd.add_input_slot(B_INPUT, READ_WRITE);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

static OpTaskSignature get_bwd_task_signature() {
  OpTaskSignature bwd(OpTaskType::BWD);

  bwd.add_input_slot(A_INPUT);
  bwd.add_input_slot(B_INPUT);
  bwd.add_input_grad_slot(A_INPUT_GRAD);
  bwd.add_input_grad_slot(B_INPUT_GRAD);
  bwd.add_output_slot(OUTPUT);
  bwd.add_output_grad_slot(OUTPUT_GRAD);

  return bwd;
}

OpTaskBinding BatchMatmul::get_init_task_binding() const {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, this->attrs);
  binding.bind_arg(PROFILING, this->profiling);

  return binding;
}

OpTaskBinding BatchMatmul::get_fwd_task_binding() const {
  OpTaskBinding binding;

  binding.bind(A_INPUT, input_tensor(0));
  binding.bind(B_INPUT, input_tensor(1));
  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(ATTRS, this->attrs);
  return binding;
}

OpTaskBinding BatchMatmul::get_bwd_task_binding() const {
  OpTaskBinding binding;
  binding.bind(A_INPUT, input_tensor(0));
  binding.bind(B_INPUT, input_tensor(1));
  binding.bind_grad(A_INPUT_GRAD, input_tensor(0).grad());
  binding.bind_grad(B_INPUT_GRAD, input_tensor(1).grad());

  binding.bind(OUTPUT, output_tensor(0));
  binding.bind_grad(OUTPUT_GRAD, output_tensor(0).grad());

  binding.bind_arg(ATTRS, this->attrs);
  return binding;
}

void BatchMatmul::init(FFModel const &ff) {
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    // init_with_dim<DIM>(ff);
    this->execute_task(ff, BATCHMATMUL_INIT_TASK_ID, get_init_task_signature());
    break;
  }
  LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
  default:
    assert(false);
}
} // namespace FlexFlow
// /
// template <int NDIM>
// void BatchMatmul::init_with_dim(FFModel const &ff) {
//   assert(check_output_input_weight_same_parallel_is());
//   parallel_is = outputs[0]->parallel_is;
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_init(ff, argmap);
//   IndexLauncher launcher(BATCHMATMUL_INIT_TASK_ID,
//                          parallel_is,
//                          TaskArgument(this, sizeof(BatchMatmul)),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     WRITE_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region));
//   launcher.add_field(0, FID_DATA);
//   for (int i = 0; i < numInputs; i++) {
//     launcher.add_region_requirement(RegionRequirement(inputs[i]->part,
//                                                       0 /*projection id*/,
//                                                       READ_ONLY,
//                                                       EXCLUSIVE,
//                                                       inputs[i]->region));
//     launcher.add_field(i + 1, FID_DATA);
//   }
//   FutureMap fm = runtime->execute_index_space(ctx, launcher);
//   fm.wait_all_results();
//   set_opmeta_from_futuremap(ff, fm);
// }

PerDeviceOpState *
    BatchMatmul::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  auto const &attrs = acc.get_argument<BatchMatmulAttrs>(ATTRS);
  bool profiling = acc.get_argument<bool>(PROFILING);

  FFHandler handle = *((FFHandler const *)task->local_args);
  BatchMatmulPerDeviceState *m = new BatchMatmulPerDeviceState(handle);
  m->profiling = profiling;
  m->a_seq_length_dim = attrs.a_seq_length_dim;
  m->b_seq_length_dim = attrs.b_seq_length_dim;
  return m;
}

void BatchMatmul::forward(FFModel const &ff) {
  int dim = outputs[0]->num_dims;
  switch (dim) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    // forward_with_dim<DIM>(ff);
    this->execute_task(ff, BATCHMATMUL_FWD_TASK_ID, get_fwd_task_signature());
    break;
  }
  LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
  default:
    assert(false);
}
}

// template <int NDIM>
// void BatchMatmul::forward_with_dim(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_forward(ff, argmap);
//   IndexLauncher launcher(
//       BATCHMATMUL_FWD_TASK_ID,
//       parallel_is,
//       TaskArgument(&ff.iter_config, sizeof(FFIterationConfig)),
//       argmap,
//       Predicate::TRUE_PRED,
//       false /*must*/,
//       0 /*mapper_id*/,
//       outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     WRITE_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region));
//   launcher.add_field(0, FID_DATA);
//   for (int i = 0; i < numInputs; i++) {
//     launcher.add_region_requirement(RegionRequirement(inputs[i]->part,
//                                                       0 /*projection id*/,
//                                                       READ_ONLY,
//                                                       EXCLUSIVE,
//                                                       inputs[i]->region));
//     launcher.add_field(i + 1, FID_DATA);
//   }
//   runtime->execute_index_space(ctx, launcher);
// }

/*
  regions[0](O): output
  regions[1](I): A
  regions[2](I): B
  ////////////////////(optional) regions[3](I): C -- TODO: is C deprecated?
  output = A * B  /////////+ C
*/
void BatchMatmul::forward_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  TaskArgumentAccessor acc(task, regions, ctx, runtime);

  // const BatchMatmul* bmm = (const BatchMatmul*) task->args;
  FFIterationConfig const *iter_config = (FFIterationConfig const *)task->args;
  // BatchMatmulMeta const *meta = *((BatchMatmulMeta **)task->local_args);
  BatchMatmulPerDeviceState const *meta =
      *((BatchMatmulPerDeviceState **)task->local_args);

  auto a_input = acc.get_tensor<READ_ONLY>(A_INPUT);
  auto b_input = acc.get_tensor<READ_ONLY>(B_INPUT);
  auto output = acc.get_tensor<WRITE_ONLY>(OUTPUT);

  int m = b_input.shape[0];
  assert(m == output.shape[0]);
  int n = a_input.shape[1];
  assert(n == output.shape[1]);
  int k = a_input.shape[0];
  assert(k == b_input.shape[1]);

  assert(a_input.shape.size() == b_input.shape.size());
  assert(a_input.shape.size() == output.shape.size());
  int batch = 1;
  for (int i = 2; i < a_input.shape.size(); i++) {
    int dim_size = a_input.shape[i];
    assert(dim_size == b_input.shape[i]);
    assert(dim_size == output.shape[i]);
    batch *= dim_size;
  }
  float *out_ptr = output.get_float_ptr();
  c float const *a_ptr = a_input.get_float_ptr();
  float const *b_ptr = b_input.get_float_ptr();
  float const *c_ptr = NULL;
  // if (regions.size() == 4) {
  //   Domain c_domain = runtime->get_index_space_domain(
  //       ctx, task->regions[3].region.get_index_space());
  //   assert(c_domain == a_domain);
  //   c_ptr = helperGetTensorPointerRO<float>(
  //       regions[3], task->regions[3], FID_DATA, ctx, runtime);
  // }

  profile(forward_kernel,
          meta->profiling,
          "[BatchMatmul] forward_time = %.2lfms\n",
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
// template <int NDIM>
// void BatchMatmul::backward_with_dim(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_backward(ff, argmap);
//   IndexLauncher launcher(
//       BATCHMATMUL_BWD_TASK_ID,
//       parallel_is,
//       TaskArgument(&ff.iter_config, sizeof(FFIterationConfig)),
//       argmap,
//       Predicate::TRUE_PRED,
//       false /*must*/,
//       0 /*mapper_id*/,
//       outputs[0]->machine_view.hash());
//   // regions[0](I): output
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region));
//   launcher.add_field(0, FID_DATA);
//   // regions[1](I): output_grad
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region_grad));
//   launcher.add_field(1, FID_DATA);
//   // regions[2](I): A
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region));
//   launcher.add_field(2, FID_DATA);
//   // regions[3](I/O): A_grad
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_WRITE,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region_grad));
//   launcher.add_field(3, FID_DATA);
//   // regions[4](I): B
//   launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[1]->region));
//   launcher.add_field(4, FID_DATA);
//   // regions[5](I/O): B_grad
//   launcher.add_region_requirement(RegionRequirement(inputs[1]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_WRITE,
//                                                     EXCLUSIVE,
//                                                     inputs[1]->region_grad));
//   launcher.add_field(5, FID_DATA);
//   runtime->execute_index_space(ctx, launcher);
// }

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
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  FFIterationConfig const *iter_config = (FFIterationConfig const *)task->args;
  BatchMatmulPerDeviceState const *meta =
      *((BatchMatmulPerDeviceState **)task->local_args);
  // output domains
  auto output = acc.get_tensor<READ_ONLY>(OUTPUT);
  auto output_grad = acc.get_tensor<READ_WRITE>(OUTPUT_GRAD);
  assert(output ==
         output_grad); // is this equivalent to checking `Domain` equality?
  // A domains
  auto a_input = acc.get_tensor<READ_ONLY>(A_INPUT);
  auto a_input_grad = acc.get_tensor<READ_WRITE>(A_INPUT_GRAD);
  assert(a_input == a_input_grad);
  // B domains
  auto b_input = acc.get_tensor<READ_ONLY>(B_INPUT);
  auto b_input_grad = acc.get_tensor<READ_WRITE>(B_INPUT_GRAD);
  assert(b_input == b_input_grad);

  // check dins
  int m = b_input.shape[0];
  assert(m == output.shape[0]);
  int n = a_input.shape[1];
  assert(n == output.shape[1]);
  int k = a_input.shape[0];
  assert(k == b_input.shape[1]);
  assert(a_input.shape.size() == b_input.shape.size());
  assert(a_input.shape.size() == output.shape.size());
  int batch = 1;
  for (int i = 2; i < a_input.shape.size(); i++) {
    int dim_size = a_input.shape[i];
    assert(dim_size == b_input.shape[i]);
    assert(dim_size == output.shape[i]);
    batch *= dim_size;
  }
  // get pointers
  float const *out_ptr = output.get_float_ptr();
  float const *out_grad_ptr = output_grad.get_float_ptr();
  float const *a_ptr = a_input.get_float_ptr();
  float *a_grad_ptr = a_input_grad.get_float_ptr();
  float const *b_ptr = b_input.get_float_ptr();
  float *b_grad_ptr = b_input_grad.get_float_ptr();

  float *c_grad_ptr = NULL;

  // TODO: add support for meta->a_seq_length_dim >= 0
  // or meta->b_seq_length_dim >= 0
  assert((meta->a_seq_length_dim >= a_len) || (iter_config->seq_length == 0));
  assert((meta->b_seq_length_dim >= b_len) || (iter_config->seq_length == 0));

  profile(backward_kernel,
          meta->profiling,
          "[BatchMatmul] backward_time = %.2lfms\n",
          meta,
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

  BatchMatmulPerDeviceState *meta = sim->batch_matmul_meta;

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
  forward = [&](ffStream_t stream) {
    forward_kernel(stream, meta, out_ptr, a_ptr, b_ptr, c_ptr, m, n, k, batch);
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

    backward = [&](ffStream_t stream) {
      backward_kernel(stream,
                      meta,
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
}
; // namespace FlexFlow

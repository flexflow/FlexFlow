#ifndef _FLEXFLOW_BATCH_MATMUL_H
#define _FLEXFLOW_BATCH_MATMUL_H

// #include "op-attrs/ops/batch_matmul.h"
// #include "task_spec/op_task_invocation.h"
// #include "task_spec/op_task_signature.h"
// #include "sim_environment.h"

#include "op-attrs/ops/batch_matmul.h"
#include "sim_environment.h"
#include "task_spec/op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<BATCHMATMUL_INIT_TASK_ID>();
template <>
void register_task<BATCHMATMUL_FWD_TASK_ID>();
template <>
void register_task<BATCHMATMUL_BWD_TASK_ID>();

OpTaskInvocation init(BatchMatmulAttrs const &);
OpTaskInvocation forward(BatchMatmulAttrs const &);
OpTaskInvocation backward(BatchMatmulAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  BatchMatmulAttrs const &attrs,
                                  InputParallelTensorDesc const &a_input,
                                  InputParallelTensorDesc const &b_input,
                                  ProfilingSettings const &settings,
                                  MachineView const &pc);

} // namespace FlexFlow

#endif

// BatchMatmulParams BatchMatmul::get_params() const {
//   BatchMatmulParams params;
//   params.a_seq_length_dim = inputs[0]->num_dims - 1 - this->a_seq_length_dim;
//   params.b_seq_length_dim = inputs[1]->num_dims - 1 - this->b_seq_length_dim;
//   return params;
// }

// Tensor FFModel::batch_matmul(const Tensor A,
//                              const Tensor B,
//                              int a_seq_length_dim,
//                              int b_seq_length_dim,
//                              char const *name) {
//   Layer *bmm = new Layer(this,
//                          OP_BATCHMATMUL,
//                          DT_FLOAT,
//                          name,
//                          2 /*inputs*/,
//                          0 /*weights*/,
//                          1 /*outputs*/,
//                          A,
//                          B);
//   assert((a_seq_length_dim <= 1) &&
//          "FlexFlow currently only supports seq_length_dim of 0 or 1 (in "
//          "Fortran ordering).");
//   assert((b_seq_length_dim <= 1) &&
//          "FlexFlow currently only supports seq_length_dim of 0 or 1 (in "
//          "Fortran ordering).");
//   assert(A->num_dims == B->num_dims);
//   for (int i = A->num_dims - 1; i >= 2; i--) {
//     assert(A->dims[i] == B->dims[i]);
//   }
//   assert(A->dims[0] == B->dims[1]);
//   int dims[MAX_TENSOR_DIM];
//   int numdim = A->num_dims;
//   for (int i = 0; i < numdim; i++) {
//     dims[i] = A->dims[i];
//   }
//   dims[0] = B->dims[0];
//   bmm->outputs[0] = create_tensor_legion_ordering(
//       numdim, dims, A->data_type, bmm, 0, true /*create_grad*/);
//   bmm->add_int_property("a_seq_length_dim", a_seq_length_dim);
//   bmm->add_int_property("b_seq_length_dim", b_seq_length_dim);
//   layers.push_back(bmm);
//   return bmm->outputs[0];
// }

// Op *BatchMatmul::create_operator_from_layer(
//     FFModel &model,
//     Layer const *layer,
//     std::vector<ParallelTensor> const &inputs) {
//   long long value;
//   layer->get_int_property("a_seq_length_dim", value);
//   int a_seq_length_dim = value;
//   layer->get_int_property("b_seq_length_dim", value);
//   int b_seq_length_dim = value;
//   return new BatchMatmul(model,
//                          inputs[0],
//                          inputs[1],
//                          a_seq_length_dim,
//                          b_seq_length_dim,
//                          layer->name);
// }

// BatchMatmul::BatchMatmul(
//     FFModel &model,
//     BatchMatmulParams const &params,
//     std::pair<ParallelTensor, ParallelTensor> const &inputs,
//     char const *name)
//     : BatchMatmul(model,
//                   inputs.first,
//                   inputs.second,
//                   params.a_seq_length_dim,
//                   params.b_seq_length_dim,
//                   name) {}

// // return A*B
// BatchMatmul::BatchMatmul(FFModel &model,
//                          const ParallelTensor A,
//                          const ParallelTensor B,
//                          int _a_seq_length_dim,
//                          int _b_seq_length_dim,
//                          char const *name)
//     : Op(model,
//          OP_BATCHMATMUL,
//          DT_FLOAT,
//          name,
//          2 /*inputs*/,
//          0 /*weights*/,
//          1 /*outputs*/,
//          A,
//          B),
//       a_seq_length_dim(A->num_dims - 1 - _a_seq_length_dim),
//       b_seq_length_dim(B->num_dims - 1 - _b_seq_length_dim) {
//   assert((_a_seq_length_dim <= 1) &&
//          "FlexFlow currently only supports seq_length_dim of 0 or 1 (in "
//          "Fortran ordering).");
//   assert((_b_seq_length_dim <= 1) &&
//          "FlexFlow currently only supports seq_length_dim of 0 or 1 (in "
//          "Fortran ordering).");
//   assert(A->num_dims == B->num_dims);
//   for (int i = A->num_dims - 1; i >= 2; i--) {
//     assert(A->dims[i] == B->dims[i]);
//   }
//   assert(A->dims[0] == B->dims[1]);
//   ParallelDim dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < A->num_dims; i++) {
//     dims[i] = A->dims[i];
//   }
//   dims[0] = B->dims[0];
//   numOutputs = 1;
//   outputs[0] = model.create_parallel_tensor_legion_ordering(
//       A->num_dims, dims, DT_FLOAT, this);
//   // C is not none
//   // if (C != Tensor::NO_TENSOR) {
//   //  numInputs = 3;
//   //  assert(C.num_dims == outputs[0].num_dims);
//   //  for (int i = 0; i < C.num_dims; i++)
//   //    assert(C.adim[i] == outputs[0].adim[i]);
//   //}
// }

// void BatchMatmul::serialize(Legion::Serializer &sez) const {
//   BatchMatmulParams params = get_params();
//   sez.serialize(params.a_seq_length_dim);
//   sez.serialize(params.b_seq_length_dim);
// }

// using PCG::Node;
// /*static*/
// Node BatchMatmul::deserialize(FFModel &ff,
//                               Legion::Deserializer &dez,
//                               ParallelTensor inputs[],
//                               int num_inputs) {
//   assert(num_inputs == 2);
//   int a_seq_length_dim, b_seq_length_dim;
//   dez.deserialize(a_seq_length_dim);
//   dez.deserialize(b_seq_length_dim);

//   BatchMatmulParams params;
//   params.a_seq_length_dim = a_seq_length_dim;
//   params.b_seq_length_dim = b_seq_length_dim;
//   return ff.get_or_create_node<BatchMatmul>({inputs[0], inputs[1]}, params);
// }

// Op *BatchMatmul::materialize(FFModel &ff,
//                              ParallelTensor inputs[],
//                              int num_inputs) const {
//   BatchMatmulParams params = get_params();
//   return new BatchMatmul(ff, params, {inputs[0], inputs[1]}, this->name);
// }

// void BatchMatmul::forward(FFModel const &ff) {
//   int dim = outputs[0]->num_dims;
//   switch (dim) {
// #define DIMFUNC(DIM) \
//   case DIM: { \
//     // forward_with_dim<DIM>(ff);
//     this->execute_task(ff, BATCHMATMUL_FWD_TASK_ID,
//     get_fwd_task_signature()); break;
//   }
//   LEGION_FOREACH_N(DIMFUNC)
// #undef DIMFUNC
//   default:
//     assert(false);
// }
// }

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

// void BatchMatmul::init(FFModel const &ff) {
//   int dim = outputs[0]->num_dims;
//   switch (dim) {
// #define DIMFUNC(DIM) \
//   case DIM: { \
//     // init_with_dim<DIM>(ff);
//     this->execute_task(ff, BATCHMATMUL_INIT_TASK_ID,
//     get_init_task_signature()); break;
//   }
//   LEGION_FOREACH_N(DIMFUNC)
// #undef DIMFUNC
//   default:
//     assert(false);
// }
// } // namespace FlexFlow
// // /
// // template <int NDIM>
// // void BatchMatmul::init_with_dim(FFModel const &ff) {
// //   assert(check_output_input_weight_same_parallel_is());
// //   parallel_is = outputs[0]->parallel_is;
// //   ArgumentMap argmap;
// //   Context ctx = ff.config.lg_ctx;
// //   Runtime *runtime = ff.config.lg_hlr;
// //   set_argumentmap_for_init(ff, argmap);
// //   IndexLauncher launcher(BATCHMATMUL_INIT_TASK_ID,
// //                          parallel_is,
// //                          TaskArgument(this, sizeof(BatchMatmul)),
// //                          argmap,
// //                          Predicate::TRUE_PRED,
// //                          false /*must*/,
// //                          0 /*mapper_id*/,
// //                          outputs[0]->machine_view.hash());
// //   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
// //                                                     0 /*projection id*/,
// //                                                     WRITE_ONLY,
// //                                                     EXCLUSIVE,
// //                                                     outputs[0]->region));
// //   launcher.add_field(0, FID_DATA);
// //   for (int i = 0; i < numInputs; i++) {
// //     launcher.add_region_requirement(RegionRequirement(inputs[i]->part,
// //                                                       0 /*projection id*/,
// //                                                       READ_ONLY,
// //                                                       EXCLUSIVE,
// //                                                       inputs[i]->region));
// //     launcher.add_field(i + 1, FID_DATA);
// //   }
// //   FutureMap fm = runtime->execute_index_space(ctx, launcher);
// //   fm.wait_all_results();
// //   set_opmeta_from_futuremap(ff, fm);
// // }

// OpTaskBinding BatchMatmul::get_bwd_task_binding() const {
//   OpTaskBinding binding;
//   binding.bind(A_INPUT, input_tensor(0));
//   binding.bind(B_INPUT, input_tensor(1));
//   binding.bind_grad(A_INPUT_GRAD, input_tensor(0).grad());
//   binding.bind_grad(B_INPUT_GRAD, input_tensor(1).grad());

//   binding.bind(OUTPUT, output_tensor(0));
//   binding.bind_grad(OUTPUT_GRAD, output_tensor(0).grad());

//   binding.bind_arg(ATTRS, this->attrs);
//   return binding;
// }

// static OpTaskSignature get_fwd_task_signature() {
//   OpTaskSignature fwd(OpTaskType::FWD);

//   fwd.add_input_slot(A_INPUT, READ_WRITE);
//   fwd.add_input_slot(B_INPUT, READ_WRITE);
//   fwd.add_output_slot(OUTPUT);

//   return fwd;
// }

// static OpTaskSignature get_bwd_task_signature() {
//   OpTaskSignature bwd(OpTaskType::BWD);

//   bwd.add_input_slot(A_INPUT);
//   bwd.add_input_slot(B_INPUT);
//   bwd.add_input_grad_slot(A_INPUT_GRAD);
//   bwd.add_input_grad_slot(B_INPUT_GRAD);
//   bwd.add_output_slot(OUTPUT);
//   bwd.add_output_grad_slot(OUTPUT_GRAD);

//   return bwd;
// }

// OpTaskBinding BatchMatmul::get_init_task_binding() const {
//   OpTaskBinding binding;

//   binding.bind_arg(ATTRS, this->attrs);
//   binding.bind_arg(PROFILING, this->profiling);

//   return binding;
// }

// OpTaskBinding BatchMatmul::get_fwd_task_binding() const {
//   OpTaskBinding binding;

//   binding.bind(A_INPUT, input_tensor(0));
//   binding.bind(B_INPUT, input_tensor(1));
//   binding.bind(OUTPUT, output_tensor(0));

//   binding.bind_arg(ATTRS, this->attrs);
//   return binding;
// }

// void BatchMatmul::backward(FFModel const &ff) {
//    int dim = outputs[0]->num_dims;
//    switch (dim) {
//  #d ef ine DIMFUNC(DIM) \
//   case DIM: { \
//     backward_with_dim<DIM>(ff); \
//     break; \
//   }
//      LEGION_FOREACH_N(DIMFUNC)
//  #undef DIMFUNC
//      default:
//        assert(false);
//    }
//  }

// void BatchMatmul::print_layer(FFModel const &ff) {
//   return;
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

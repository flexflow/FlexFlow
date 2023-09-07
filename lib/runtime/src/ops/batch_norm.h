#ifndef _FLEXFLOW_BATCH_NORM_H
#define _FLEXFLOW_BATCH_NORM_H

#include "op-attrs/ops/batch_norm.h"
#include "sim_environment.h"
#include "task_spec/op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<BATCHNORM_INIT_TASK_ID>();
template <>
void register_task<BATCHNORM_FWD_TASK_ID>();
template <>
void register_task<BATCHNORM_BWD_TASK_ID>();

OpTaskInvocation init(BatchNormAttrs const &);
OpTaskInvocation forward(BatchNormAttrs const &);
OpTaskInvocation backward(BatchNormAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  BatchNormAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class BatchNorm { */
/* public: */
/*   BatchNorm(FFModel &model, */
/*             const ParallelTensor input, */
/*             const ParallelTensor scale, */
/*             const ParallelTensor bias, */
/*             bool relu, */
/*             char const *name); */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */
/*   void update(FFModel const &); */

/*   static PerDeviceOpState *init_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void forward_task(Legion::Task const *task, */
/*                            std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                            Legion::Context ctx, */
/*                            Legion::Runtime *runtime); */
/*   static void backward_task(Legion::Task const *task, */
/*                             std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                             Legion::Context ctx, */
/*                             Legion::Runtime *runtime); */
/*   bool measure_operator_cost(Simulator *sim, */
/*                              MachineView const &pc, */
/*                              CostMetrics &cost_metrics) const override; */
/*   OpTaskBinding get_init_task_binding() const override; */
/*   OpTaskBinding get_fwd_task_binding() const override; */
/*   OpTaskBinding get_bwd_task_binding() const override; */
/* public: */
/*   bool relu; */
/*   int num_replica; */
/* }; */

} // namespace FlexFlow

#endif

// void BatchNorm::init(FFModel const &ff) {
//   this->execute_task(ff, BATCHNORM_INIT_TASK_ID, get_init_task_signature());
// }

// void BatchNorm::forward(FFModel const &ff) {
//   this->execute_task(ff, BATCHNORM_FWD_TASK_ID, get_fwd_task_signature());
// }

// void BatchNorm::backward(FFModel const &ff) {
//   this->execute_task(ff, BATCHNORM_BWD_TASK_ID, get_bwd_task_signature());
// }

// Tensor batch_norm(const Tensor input, bool relu, char const *name) {
//   assert(input->num_dims == 4); /*NCHW*/
//   Layer *bm = new Layer(this,
//                         OP_BATCHNORM,
//                         DT_FLOAT,
//                         name,
//                         1 /*inputs*/,
//                         2 /*weights*/,
//                         1 /*outputs*/,
//                         input);
//   int numdims = 4;
//   bm->outputs[0] = create_tensor_legion_ordering(
//       numdims, input->dims, DT_FLOAT, bm, 0, true /*create_grad*/);
//   bm->add_int_property("relu", relu);
//   layers.push_back(bm);
//   return bm->outputs[0];
// }

// BatchNorm::BatchNorm(FFModel &model,
//                      const ParallelTensor _input,
//                      const ParallelTensor _scale,
//                      const ParallelTensor _bias,
//                      bool _relu,
//                      char const *name)
//     : Op(model,
//          OP_BATCHNORM,
//          DT_FLOAT,
//          name,
//          1 /*inputs*/,
//          2 /*weights*/,
//          1 /*outputs*/,
//          _input,
//          _scale,
//          _bias),
//       relu(_relu) {
//   assert(_input->num_dims == 4);
//   numOutputs = 1;
//   ParallelDim dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < _input->num_dims; i++) {
//     dims[i] = _input->dims[_input->num_dims - 1 - i];
//   }
//   outputs[0] =
//       model.create_parallel_tensor(_input->num_dims, dims, DT_FLOAT, this);
//   return;
// }

/*
  locals[0] = scale
  locals[1] = bias
*/

// void BatchNorm::init(FFModel const &ff) {
//   this->execute_task(ff, BATCHNORM_INIT_TASK_ID, get_init_task_signature());
// assert(check_output_input_weight_same_parallel_is());
// parallel_is = outputs[0]->parallel_is;
// ArgumentMap argmap;
// Context ctx = ff.config.lg_ctx;
// Runtime *runtime = ff.config.lg_hlr;
// set_argumentmap_for_init(ff, argmap);
// IndexLauncher launcher(BATCHNORM_INIT_TASK_ID,
//                        parallel_is,
//                        TaskArgument(this, sizeof(BatchNorm)),
//                        argmap,
//                        Predicate::TRUE_PRED,
//                        false /*must*/,
//                        0 /*mapper_id*/,
//                        outputs[0]->machine_view.hash());
// launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   inputs[0]->region));
// launcher.add_field(0, FID_DATA);
// launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                   0 /*projection id*/,
//                                                   WRITE_ONLY,
//                                                   EXCLUSIVE,
//                                                   outputs[0]->region));
// launcher.add_field(1, FID_DATA);
// launcher.add_region_requirement(RegionRequirement(weights[0]->region,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   weights[0]->region));
// launcher.add_field(2, FID_DATA);
// launcher.add_region_requirement(RegionRequirement(weights[1]->region,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   weights[1]->region));
// launcher.add_field(3, FID_DATA);
// FutureMap fm = runtime->execute_index_space(ctx, launcher);
// fm.wait_all_results();
// set_opmeta_from_futuremap(ff, fm);
//   }

/*
  regions[0]: input
  regions[1]: output
  regions[2](I): scale
  regions[3](I): bias
*/

// void BatchNorm::forward(FFModel const &ff) {
//   this->execute_task(ff, BATCHNORM_FWD_TASK_ID, get_fwd_task_signature());
// ArgumentMap argmap;
// Context ctx = ff.config.lg_ctx;
// Runtime *runtime = ff.config.lg_hlr;
// set_argumentmap_for_forward(ff, argmap);
// IndexLauncher launcher(BATCHNORM_FWD_TASK_ID,
//                        parallel_is,
//                        TaskArgument(NULL, 0),
//                        argmap,
//                        Predicate::TRUE_PRED,
//                        false /*must*/,
//                        0 /*mapper_id*/,
//                        outputs[0]->machine_view.hash());
// launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   inputs[0]->region));
// launcher.add_field(0, FID_DATA);
// launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                   0 /*projection id*/,
//                                                   WRITE_DISCARD,
//                                                   EXCLUSIVE,
//                                                   outputs[0]->region));
// launcher.add_field(1, FID_DATA);
// launcher.add_region_requirement(RegionRequirement(weights[0]->region,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   weights[0]->region));
// launcher.add_field(2, FID_DATA);
// launcher.add_region_requirement(RegionRequirement(weights[1]->region,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   weights[1]->region));
// launcher.add_field(3, FID_DATA);

// runtime->execute_index_space(ctx, launcher);
// }

/*
  regions[0](I): input
  regions[1](O): ouptut
  regions[2](I): scale
  regions[3](I): bias
*/

// void BatchNorm::backward(FFModel const &ff) {
//   this->execute_task(ff, BATCHNORM_BWD_TASK_ID, get_bwd_task_signature());
// ArgumentMap argmap;
// Context ctx = ff.config.lg_ctx;
// Runtime *runtime = ff.config.lg_hlr;
// set_argumentmap_for_backward(ff, argmap);
// IndexLauncher launcher(BATCHNORM_BWD_TASK_ID,
//                        parallel_is,
//                        TaskArgument(NULL, 0),
//                        argmap,
//                        Predicate::TRUE_PRED,
//                        false /*must*/,
//                        0 /*mapper_id*/,
//                        outputs[0]->machine_view.hash());
// // regions[0](I): input
// launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   inputs[0]->region));
// launcher.add_field(0, FID_DATA);
// // regions[1](I/O): input_grad (we only need grad tensors)
// launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
//                                                   0 /*projection id*/,
//                                                   READ_WRITE,
//                                                   EXCLUSIVE,
//                                                   inputs[0]->region_grad));
// launcher.add_field(1, FID_DATA);
// // regions[2](I): output
// launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   outputs[0]->region));
// launcher.add_field(2, FID_DATA);
// // regions[3](I/O): output_grad
// launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
//                                                   0 /*projection id*/,
//                                                   READ_WRITE,
//                                                   EXCLUSIVE,
//                                                   outputs[0]->region_grad));
// launcher.add_field(3, FID_DATA);
// // regions[4](I): filter
// launcher.add_region_requirement(RegionRequirement(weights[0]->region,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   weights[0]->region));
// launcher.add_field(4, FID_DATA);
// // regions[5](I/O): filter_grad
// launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
//                                                   0 /*projection id*/,
//                                                   READ_WRITE,
//                                                   EXCLUSIVE,
//                                                   weights[0]->region_grad));
// launcher.add_field(5, FID_DATA);
// // regions[6](I/O): bias_grad
// launcher.add_region_requirement(RegionRequirement(weights[1]->part_grad,
//                                                   0 /*projection id*/,
//                                                   READ_WRITE,
//                                                   EXCLUSIVE,
//                                                   weights[1]->region_grad));
// launcher.add_field(6, FID_DATA);
// FutureMap fm = runtime->execute_index_space(ctx, launcher);
// }

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): scale
  regions[5](I/O): scale_grad
  regions[6](I/O): bias_grad
*/
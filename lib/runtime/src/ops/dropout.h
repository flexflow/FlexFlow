#ifndef _FLEXFLOW_DROPOUT_H
#define _FLEXFLOW_DROPOUT_H

#include "op-attrs/ops/dropout.h"
#include "task_spec/op_task_invocation.h"
#include "sim_environment.h"
#include "tasks.h"

namespace FlexFlow {

template <>
void register_task<DROPOUT_INIT_TASK_ID>();
template <>
void register_task<DROPOUT_FWD_TASK_ID>();
template <>
void register_task<DROPOUT_BWD_TASK_ID>();

OpTaskInvocation init(DropoutAttrs const &);
OpTaskInvocation forward(DropoutAttrs const &);
OpTaskInvocation backward(DropoutAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  DropoutAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Dropout : public Op { */
/* public: */
/*   Dropout(FFModel &model, */
/*           const ParallelTensor input, */
/*           float rate, */
/*           unsigned long long seed, */
/*           char const *name); */
/*   Dropout(FFModel &model, Dropout const &other, const ParallelTensor input);
 */
/*   void init(FFModel const &) override; */
/*   void forward(FFModel const &) override; */
/*   void backward(FFModel const &) override; */

/*   static Op * */
/*       create_operator_from_layer(FFModel &model, */
/*                                  Layer const *layer, */
/*                                  std::vector<ParallelTensor> const &inputs);
 */

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

/*   /1* static PCG::Node deserialize(FFModel &ff, *1/ */
/*   /1*                              Legion::Deserializer &d, *1/ */
/*   /1*                              ParallelTensor inputs[], *1/ */
/*   /1*                              int num_inputs); *1/ */

/* public: */
/*   float rate; */
/*   unsigned long long seed; */
/* }; */

} // namespace FlexFlow

#endif



// Tensor FFModel::dropout(const Tensor input,
//                         float rate,
//                         unsigned long long seed,
//                         char const *name) {
//   // seed = 0 is preserved as None, so we use a random seed
//   if (seed == 0) {
//     seed = std::rand();
//   }
//   Layer *dropout = new Layer(this,
//                              OP_DROPOUT,
//                              DT_FLOAT,
//                              name,
//                              1 /*inputs*/,
//                              0 /*weights*/,
//                              1 /*outputs*/,
//                              input);
//   int numdims = input->num_dims;
//   int dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < numdims; i++) {
//     dims[i] = input->dims[i];
//   }
//   dropout->outputs[0] = create_tensor_legion_ordering(
//       numdims, dims, DT_FLOAT, dropout, 0, true /*create_grad*/);
//   dropout->add_float_property("rate", rate);
//   dropout->add_int_property("seed", seed);
//   layers.push_back(dropout);
//   return dropout->outputs[0];
// }

// Op *Dropout::create_operator_from_layer(
//     FFModel &model,
//     Layer const *layer,
//     std::vector<ParallelTensor> const &inputs) {
//   long long value;
//   layer->get_int_property("seed", value);
//   int seed = value;
//   float rate;
//   layer->get_float_property("rate", rate);
//   return new Dropout(model, inputs[0], rate, seed, layer->name);
// }

// DropoutParams Dropout::get_params() const {
//   DropoutParams params;
//   params.rate = this->rate;
//   params.seed = this->seed;
//   return params;
// }

// Dropout::Dropout(FFModel &model,
//                  const ParallelTensor _input,
//                  float _rate,
//                  unsigned long long _seed,
//                  char const *name)
//     : Op(model,
//          OP_DROPOUT,
//          DT_FLOAT,
//          name,
//          1 /*inputs*/,
//          0 /*weights*/,
//          1 /*outputs*/,
//          _input),
//       rate(_rate), seed(_seed) {
//   // Set output shape
//   ParallelDim dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < _input->num_dims; i++) {
//     dims[i] = _input->dims[i];
//   }
//   numOutputs = 1;
//   outputs[0] = model.create_parallel_tensor_legion_ordering(
//       _input->num_dims, dims, DT_FLOAT, this);
// }

// Dropout::Dropout(FFModel &model,
//                  Dropout const &other,
//                  const ParallelTensor input)
//     : Dropout(model, input, other.rate, other.seed, other.name) {}

// Dropout::Dropout(FFModel &model,
//                  DropoutParams const &params,
//                  const ParallelTensor input,
//                  char const *name)
//     : Dropout(model, input, params.rate, params.seed, name) {}

// // void Dropout::init(FFModel const &ff) {
// //   assert(check_output_input_weight_same_parallel_is());
// //   parallel_is = outputs[0]->parallel_is;
// //   ArgumentMap argmap;
// //   Context ctx = ff.config.lg_ctx;
// //   Runtime *runtime = ff.config.lg_hlr;
// //   set_argumentmap_for_init(ff, argmap);
// //   IndexLauncher init_launcher(DROPOUT_INIT_TASK_ID,
// //                               parallel_is,
// //                               TaskArgument(this, sizeof(Dropout)),
// //                               argmap,
// //                               Predicate::TRUE_PRED,
// //                               false /*must*/,
// //                               0 /*mapper_id*/,
// //                               outputs[0]->machine_view.hash());
// //   init_launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
// //                                                          0 /*projection id*/,
// //                                                          READ_ONLY,
// //                                                          EXCLUSIVE,
// //                                                          inputs[0]->region));
// //   init_launcher.add_field(0, FID_DATA);
// //   init_launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
// //                                                          0 /*projection id*/,
// //                                                          WRITE_ONLY,
// //                                                          EXCLUSIVE,
// //                                                          outputs[0]->region));
// //   init_launcher.add_field(1, FID_DATA);
// //   FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
// //   fm.wait_all_results();
// //   set_opmeta_from_futuremap(ff, fm);
// // }

// void Dropout::forward(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_forward(ff, argmap);
//   IndexLauncher launcher(DROPOUT_FWD_TASK_ID,
//                          parallel_is,
//                          TaskArgument(NULL, 0),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region));
//   launcher.add_field(0, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     WRITE_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region));
//   launcher.add_field(1, FID_DATA);
//   runtime->execute_index_space(ctx, launcher);
// }


// void Dropout::backward(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   set_argumentmap_for_backward(ff, argmap);
//   IndexLauncher launcher(DROPOUT_BWD_TASK_ID,
//                          parallel_is,
//                          TaskArgument(NULL, 0),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_WRITE,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region_grad));
//   launcher.add_field(0, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region_grad));
//   launcher.add_field(1, FID_DATA);
//   runtime->execute_index_space(ctx, launcher);
// }

/*
  regions[0](I/O): input_grad
  regions[1](I): output_grad
*/


// void Dropout::serialize(Legion::Serializer &sez) const {
//   sez.serialize(this->rate);
//   sez.serialize(this->seed);
// }

// Node Dropout::deserialize(FFModel &ff,
//                           Legion::Deserializer &dez,
//                           ParallelTensor inputs[],
//                           int num_inputs) {
//   assert(num_inputs == 1);
//   unsigned long long seed;
//   float rate;
//   dez.deserialize(rate);
//   dez.deserialize(seed);
//   DropoutParams params;
//   params.rate = rate;
//   params.seed = seed;
//   return ff.get_or_create_node<Dropout>(inputs[0], params);
// }

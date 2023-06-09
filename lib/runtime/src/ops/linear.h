#ifndef _FLEXFLOW_LINEAR_H
#define _FLEXFLOW_LINEAR_H

#include "op-attrs/ops/linear.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<LINEAR_INIT_TASK_ID>();
template <>
void register_task<LINEAR_FWD_TASK_ID>();
template <>
void register_task<LINEAR_BWD_TASK_ID>();

OpTaskInvocation init(LinearAttrs const &);
OpTaskInvocation forward(LinearAttrs const &);
OpTaskInvocation backward(LinearAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  LinearAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Linear : public Op { */
/* public: */
/*   Linear(FFModel &model, */
/*          LayerID const &layer_guid, */
/*          ParallelTensor const &input, */
/*          int out_dim, */
/*          ActiMode activation, */
/*          bool use_bias, */
/*          DataType data_type, */
/*          bool allocate_weights, */
/*          char const *name); */
/*   Linear(FFModel &model, */
/*          Linear const &other, */
/*          ParallelTensor const input, */
/*          bool allocate_weights); */
/*   Linear(FFModel &model, */
/*          LinearAttrs const &attrs, */
/*          ParallelTensor input, */
/*          char const *name = nullptr, */
/*          bool allocate_weights = false); */

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
/*   bool estimate_sync_cost(Simulator *sim, */
/*                           MachineView const &pc, */
/*                           CostMetrics &cost_metrics) const override; */
/*   /1* ParallelConfig get_random_parallel_config(FFModel const &ff) const
 * override; *1/ */
/*   /1* bool is_valid_parallel_config(FFModel const &ff, *1/ */
/*   /1*                               ParallelConfig const &pc) const override;
 * *1/ */

/*   /1* static PCG::Node deserialize(FFModel &ff, *1/ */
/*   /1*                              Legion::Deserializer &d, *1/ */
/*   /1*                              ParallelTensor inputs[], *1/ */
/*   /1*                              int num_inputs); *1/ */

/* private: */
/*   Linear(int guid, */
/*          bool profiling, */
/*          const ParallelTensor input, */
/*          int out_dim, */
/*          ActiMode activation, */
/*          bool use_bias, */
/*          bool allocate_weights, */
/*          char const *name); */

/*   template <int NDIM> */
/*   static PerDeviceOpState * */
/*       init_task_with_dim(Legion::Task const *task, */
/*                          std::vector<Legion::PhysicalRegion> const &regions,
 */
/*                          Legion::Context ctx, */
/*                          Legion::Runtime *runtime); */
/*   template <int NDIM> */
/*   static void */
/*       forward_task_with_dim(Legion::Task const *task, */
/*                             std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                             Legion::Context ctx, */
/*                             Legion::Runtime *runtime); */
/*   template <int NDIM> */
/*   static void */
/*       backward_task_with_dim(Legion::Task const *task, */
/*                              std::vector<Legion::PhysicalRegion> const
 * &regions, */
/*                              Legion::Context ctx, */
/*                              Legion::Runtime *runtime); */

/*   void register_mappings(); */
/*   void register_output_mappings(); */
/*   void register_weight_mappings(); */

/* public: */
/*   int in_channels, out_channels; */
/*   ActiMode activation; */
/*   bool use_bias; */
/* }; */

} // namespace FlexFlow

#endif

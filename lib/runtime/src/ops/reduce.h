#ifndef _FLEXFLOW_RUNTIME_SRC_OPS_REDUCE_H
#define _FLEXFLOW_RUNTIME_SRC_OPS_REDUCE_H

#include "op-attrs/ops/reduce.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<REDUCE_INIT_TASK_ID>();
template <>
void register_task<REDUCE_FWD_TASK_ID>();
template <>
void register_task<REDUCE_BWD_TASK_ID>();

OpTaskInvocation init(ReduceAttrs const &);
OpTaskInvocation forward(ReduceAttrs const &);
OpTaskInvocation backward(ReduceAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReduceAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Reduce : public Op { */
/* public: */
/*   Reduce(FFModel &model, */
/*          OperatorType op_type, */
/*          ParallelTensor const &input, */
/*          std::vector<int> const &axes, */
/*          bool keepdims, */
/*          char const *name = nullptr); */
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
/*   void serialize(Legion::Serializer &s) const override; */
/*   /1* static PCG::Node deserialize(FFModel &ff, *1/ */
/*   /1*                              Legion::Deserializer &d, *1/ */
/*   /1*                              ParallelTensor inputs[], *1/ */
/*   /1*                              int num_inputs); *1/ */
/*   Op *materialize(FFModel &ff, */
/*                   ParallelTensor inputs[], */
/*                   int num_inputs) const override; */

/* public: */
/*   stack_vector<int, MAX_TENSOR_DIM> axes; */
/*   bool keepdims; */
/* }; */

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_OPS_GATHER_H
#define _FLEXFLOW_OPS_GATHER_H

#include "op-attrs/ops/gather.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<GATHER_INIT_TASK_ID>();
template <>
void register_task<GATHER_FWD_TASK_ID>();
template <>
void register_task<GATHER_BWD_TASK_ID>();

OpTaskInvocation init(GatherAttrs const &);
OpTaskInvocation forward(GatherAttrs const &);
OpTaskInvocation backward(GatherAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  GatherAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ParallelTensorShape const &index_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Gather : public Op { */
/* public: */
/*   Gather(FFModel &model, */
/*          ParallelTensor const &input, */
/*          ParallelTensor const &index, */
/*          int legion_dim, */
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
/*   int legion_dim; */
/* }; */

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_SPLIT_H
#define _FLEXFLOW_SPLIT_H

#include "op-attrs/ops/split.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<SPLIT_INIT_TASK_ID>();
template <>
void register_task<SPLIT_FWD_TASK_ID>();
template <>
void register_task<SPLIT_BWD_TASK_ID>();

OpTaskInvocation init(SplitAttrs const &);
OpTaskInvocation forward(SplitAttrs const &);
OpTaskInvocation backward(SplitAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  SplitAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

/* class Split : public Op { */
/* public: */
/*   Split(FFModel &model, */
/*         const ParallelTensor input, */
/*         std::vector<int> const &split, */
/*         int legion_axis, */
/*         char const *name = nullptr); */
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

/* public: */
/*   int legion_axis; */
/*   std::vector<int> splits; */
/* }; */

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_OPS_GATHER_H
#define _FLEXFLOW_OPS_GATHER_H

#include "op-attrs/ops/gather.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<GATHER_FWD_TASK_ID>();
template <>
void register_task<GATHER_BWD_TASK_ID>();

OpTaskInvocation forward(GatherAttrs const &);
OpTaskInvocation backward(GatherAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  GatherAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ParallelTensorShape const &index_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);
} // namespace FlexFlow

#endif

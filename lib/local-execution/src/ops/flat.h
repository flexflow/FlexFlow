#ifndef _FLEXFLOW_FLAT_H
#define _FLEXFLOW_FLAT_H

#include "local-execution/sim_environment.h"
#include "op-attrs/ops/flat.h"

namespace FlexFlow {

template <>
void register_task<FLAT_FWD_TASK_ID>();
template <>
void register_task<FLAT_BWD_TASK_ID>();

OpTaskInvocation forward(FlatAttrs const &);
OpTaskInvocation backward(FlatAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  FlatAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);
} // namespace FlexFlow

#endif

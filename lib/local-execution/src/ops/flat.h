#ifndef _FLEXFLOW_FLAT_H
#define _FLEXFLOW_FLAT_H

#include "local-execution/sim_environment.h"
#include "op-attrs/ops/flat.h"

namespace FlexFlow {

TaskImplFunction get_flat_fwd_task_impl();
TaskImplFunction get_flat_bwd_task_impl();

OpTaskSignature get_flat_fwd_signature();
OpTaskSignature get_flat_bwd_signature();

OpTaskInvocation forward(FlatAttrs const &);
OpTaskInvocation backward(FlatAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  FlatAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);
} // namespace FlexFlow

#endif

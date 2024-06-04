#ifndef _FLEXFLOW_SPLIT_H
#define _FLEXFLOW_SPLIT_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/split.h"

namespace FlexFlow {

TaskImplFunction get_split_fwd_task_impl();
TaskImplFunction get_split_bwd_task_impl();

OpTaskSignature get_split_fwd_signature();
OpTaskSignature get_split_bwd_signature();

OpTaskInvocation forward(SplitAttrs const &);
OpTaskInvocation backward(SplitAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  SplitAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

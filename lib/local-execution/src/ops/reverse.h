#ifndef _FLEXFLOW_REVERSE_H_
#define _FLEXFLOW_REVERSE_H_

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/reverse.h"

namespace FlexFlow {

template <>
void register_task<REVERSE_FWD_TASK_ID>();
template <>
void register_task<REVERSE_BWD_TASK_ID>();

TaskImplFunction get_reverse_fwd_task_impl();
TaskImplFunction get_reverse_bwd_task_impl();

OpTaskSignature get_reverse_fwd_signature();
OpTaskSignature get_reverse_bwd_signature();

OpTaskInvocation forward(ReverseAttrs const &);
OpTaskInvocation backward(ReverseAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReverseAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

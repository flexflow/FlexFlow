#ifndef _FLEXFLOW_LINEAR_H
#define _FLEXFLOW_LINEAR_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/linear.h"

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

TaskImplFunction get_linear_init_task_impl();
TaskImplFunction get_linear_fwd_task_impl();
TaskImplFunction get_linear_bwd_task_impl();

OpTaskSignature get_linear_init_signature();
OpTaskSignature get_linear_fwd_signature();
OpTaskSignature get_linear_bwd_signature();

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  LinearAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

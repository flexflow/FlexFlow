#ifndef _FLEXFLOW_REVERSE_H_
#define _FLEXFLOW_REVERSE_H_

#include "op-attrs/ops/reverse.h"
#include "sim_environment.h"
#include "task_spec/op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<REVERSE_INIT_TASK_ID>();
template <>
void register_task<REVERSE_FWD_TASK_ID>();
template <>
void register_task<REVERSE_BWD_TASK_ID>();

OpTaskInvocation init(ReverseAttrs const &);
OpTaskInvocation forward(ReverseAttrs const &);
OpTaskInvocation backward(ReverseAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReverseAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

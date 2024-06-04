#ifndef _FLEXFLOW_COMBINE_H
#define _FLEXFLOW_COMBINE_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/combine.h"

namespace FlexFlow {

template <>
void register_task<COMBINE_FWD_TASK_ID>();
template <>
void register_task<COMBINE_BWD_TASK_ID>();

TaskImplFunction get_combine_init_task_impl();
TaskImplFunction get_combine_fwd_task_impl();
TaskImplFunction get_combine_bwd_task_impl();

OpTaskSignature get_combine_init_signature();
OpTaskSignature get_combine_fwd_signature();
OpTaskSignature get_combine_bwd_signature();

OpTaskInvocation forward(CombineAttrs const &);
OpTaskInvocation backward(CombineAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  CombineAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv);
} // namespace FlexFlow

#endif

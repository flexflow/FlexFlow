#ifndef _FLEXFLOW_REDUCTION_H
#define _FLEXFLOW_REDUCTION_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/reduction.h"

namespace FlexFlow {

template <>
void register_task<REDUCTION_FWD_TASK_ID>();
template <>
void register_task<REDUCTION_BWD_TASK_ID>();

TaskImplFunction get_reduction_fwd_task_impl();
TaskImplFunction get_reduction_bwd_task_impl();

OpTaskSignature get_reduction_fwd_signature();
OpTaskSignature get_reduction_bwd_signature();

OpTaskInvocation init(ReductionAttrs const &);
OpTaskInvocation forward(ReductionAttrs const &);
OpTaskInvocation backward(ReductionAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReductionAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

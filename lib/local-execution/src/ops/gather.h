#ifndef _FLEXFLOW_GATHER_H
#define _FLEXFLOW_GATHER_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/gather.h"

namespace FlexFlow {

template <>
void register_task<GATHER_INIT_TASK_ID>();
template <>
void register_task<GATHER_FWD_TASK_ID>();
template <>
void register_task<GATHER_BWD_TASK_ID>();

TaskImplFunction get_gather_init_task_impl();
TaskImplFunction get_gather_fwd_task_impl();
TaskImplFunction get_gather_bwd_task_impl();

OpTaskSignature get_gather_init_signature();
OpTaskSignature get_gather_fwd_signature();
OpTaskSignature get_gather_bwd_signature();

OpTaskInvocation init(GatherAttrs const &);
OpTaskInvocation forward(GatherAttrs const &);
OpTaskInvocation backward(GatherAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  GatherAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  InputParallelTensorDesc const &index,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

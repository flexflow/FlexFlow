#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/concat.h"

namespace FlexFlow {

template <>
void register_task<CONCAT_FWD_TASK_ID>();
template <>
void register_task<CONCAT_BWD_TASK_ID>();

TaskImplFunction get_concat_fwd_task_impl();
TaskImplFunction get_concat_bwd_task_impl();

OpTaskSignature get_concat_fwd_signature();
OpTaskSignature get_concat_bwd_signature();

OpTaskInvocation forward(ConcatAttrs const &);
OpTaskInvocation backward(ConcatAttrs const &);

CostMetrics
    measure_operator_cost(SimEnvFactory const &sim_factory,
                          ConcatAttrs const &attrs,
                          std::vector<ParallelTensorShape> const &input_shapes,
                          ProfilingSettings const &settings,
                          MachineView const &machine_view);
} // namespace FlexFlow

#endif

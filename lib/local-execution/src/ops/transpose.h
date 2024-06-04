#ifndef _FLEXFLOW_TRANSPOSE_H_
#define _FLEXFLOW_TRANSPOSE_H_

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/transpose.h"

namespace FlexFlow {

template <>
void register_task<TRANSPOSE_INIT_TASK_ID>();
template <>
void register_task<TRANSPOSE_FWD_TASK_ID>();
template <>
void register_task<TRANSPOSE_BWD_TASK_ID>();

TaskImplFunction get_transpose_init_task_impl();
TaskImplFunction get_transpose_fwd_task_impl();
TaskImplFunction get_transpose_bwd_task_impl();

OpTaskSignature get_transpose_init_signature();
OpTaskSignature get_transpose_fwd_signature();
OpTaskSignature get_transpose_bwd_signature();

OpTaskInvocation init(TransposeAttrs const &);
OpTaskInvocation forward(TransposeAttrs const &);
OpTaskInvocation backward(TransposeAttrs const &);

CostMetrics
    measure_operator_cost(SimEnvFactory const &sim_factory,
                          TransposeAttrs const &attrs,
                          InputVariadicParallelTensorDesc const
                              &input_descs, // Note:this may have some problem
                          ProfilingSettings const &settings,
                          MachineView const &machine_view);

} // namespace FlexFlow

#endif

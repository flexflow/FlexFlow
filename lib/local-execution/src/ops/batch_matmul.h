#ifndef _FLEXFLOW_BATCH_MATMUL_H
#define _FLEXFLOW_BATCH_MATMUL_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/op_task_signature.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/batch_matmul.h"

namespace FlexFlow {

template <>
void register_task<BATCHMATMUL_FWD_TASK_ID>();
template <>
void register_task<BATCHMATMUL_BWD_TASK_ID>();

TaskImplFunction get_batch_matmul_fwd_task_impl();
TaskImplFunction get_batch_matmul_bwd_task_impl();

OpTaskSignature get_batch_matmul_fwd_signature();
OpTaskSignature get_batch_matmul_bwd_signature();

OpTaskInvocation forward(BatchMatmulAttrs const &);
OpTaskInvocation backward(BatchMatmulAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  BatchMatmulAttrs const &attrs,
                                  InputParallelTensorDesc const &a_input,
                                  InputParallelTensorDesc const &b_input,
                                  ProfilingSettings const &settings,
                                  MachineView const &pc);

} // namespace FlexFlow

#endif

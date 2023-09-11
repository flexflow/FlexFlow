#ifndef _FLEXFLOW_BATCH_MATMUL_H
#define _FLEXFLOW_BATCH_MATMUL_H

#include "op-attrs/ops/batch_matmul.h"
#include "sim_environment.h"
#include "task_spec/op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<BATCHMATMUL_INIT_TASK_ID>();
template <>
void register_task<BATCHMATMUL_FWD_TASK_ID>();
template <>
void register_task<BATCHMATMUL_BWD_TASK_ID>();

OpTaskInvocation init(BatchMatmulAttrs const &);
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
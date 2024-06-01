#ifndef _FLEXFLOW_BATCH_NORM_H
#define _FLEXFLOW_BATCH_NORM_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/batch_norm.h"

namespace FlexFlow {

template <>
void register_task<BATCHNORM_INIT_TASK_ID>();
template <>
void register_task<BATCHNORM_FWD_TASK_ID>();
template <>
void register_task<BATCHNORM_BWD_TASK_ID>();

OpTaskInvocation init(BatchNormAttrs const &);
OpTaskInvocation forward(BatchNormAttrs const &);
OpTaskInvocation backward(BatchNormAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  BatchNormAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

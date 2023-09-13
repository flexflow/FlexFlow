#ifndef _FLEXFLOW_DROPOUT_H
#define _FLEXFLOW_DROPOUT_H

#include "op-attrs/ops/dropout.h"
#include "task_spec/op_task_invocation.h"
#include "sim_environment.h"
#include "tasks.h"

namespace FlexFlow {

template <>
void register_task<DROPOUT_INIT_TASK_ID>();
template <>
void register_task<DROPOUT_FWD_TASK_ID>();
template <>
void register_task<DROPOUT_BWD_TASK_ID>();

OpTaskInvocation init(DropoutAttrs const &);
OpTaskInvocation forward(DropoutAttrs const &);
OpTaskInvocation backward(DropoutAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  DropoutAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv);
} // namespace FlexFlow

#endif
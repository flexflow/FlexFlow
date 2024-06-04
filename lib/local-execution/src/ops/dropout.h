#ifndef _FLEXFLOW_DROPOUT_H
#define _FLEXFLOW_DROPOUT_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "local-execution/tasks.h"
#include "op-attrs/ops/dropout.h"

namespace FlexFlow {

TaskImplFunction get_dropout_init_task_impl();
TaskImplFunction get_dropout_fwd_task_impl();
TaskImplFunction get_dropout_bwd_task_impl();

OpTaskSignature get_dropout_init_signature();
OpTaskSignature get_dropout_fwd_signature();
OpTaskSignature get_dropout_bwd_signature();

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

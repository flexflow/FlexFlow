#ifndef _FLEXFLOW_COMBINE_H
#define _FLEXFLOW_COMBINE_H

#include "op-attrs/ops/combine.h"
#include "sim_environment.h"
#include "task_spec/op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<COMBINE_FWD_TASK_ID>();
template <>
void register_task<COMBINE_BWD_TASK_ID>();

OpTaskInvocation forward(CombineAttrs const &);
OpTaskInvocation backward(CombineAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  CombineAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv);
} // namespace FlexFlow

#endif

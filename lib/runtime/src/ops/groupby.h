#ifndef _FLEXFLOW_GROUPBY_H_
#define _FLEXFLOW_GROUPBY_H_

#include "op-attrs/ops/groupby.h"
#include "sim_environment.h"
#include "task_spec/op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<GROUP_BY_INIT_TASK_ID>();
template <>
void register_task<GROUP_BY_FWD_TASK_ID>();
template <>
void register_task<GROUP_BY_BWD_TASK_ID>();

OpTaskInvocation init(Group_byAttrs const &);
OpTaskInvocation forward(Group_byAttrs const &);
OpTaskInvocation backward(Group_byAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  Group_byAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ParallelTensorShape const &assign_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);
} // namespace FlexFlow

#endif

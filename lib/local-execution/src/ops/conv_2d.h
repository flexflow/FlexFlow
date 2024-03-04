#ifndef _FLEXFLOW_CONV_2D_H
#define _FLEXFLOW_CONV_2D_H

#include "op-attrs/ops/conv_2d.h"
#include "sim_environment.h"
#include "task_spec/op_task_invocation.h"

namespace FlexFlow {

template <>
void register_task<CONV2D_INIT_TASK_ID>();
template <>
void register_task<CONV2D_FWD_TASK_ID>();
template <>
void register_task<CONV2D_BWD_TASK_ID>();

OpTaskInvocation init(Conv2DAttrs const &);
OpTaskInvocation forward(Conv2DAttrs const &);
OpTaskInvocation backward(Conv2DAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  Conv2DAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_ELEMENT_BINARY_H
#define _FLEXFLOW_ELEMENT_BINARY_H

#include "op-attrs/ops/element_binary.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<ELEMENTBINARY_INIT_TASK_ID>();

template <>
void register_task<ELEMENTBINARY_FWD_TASK_ID>();

template <>
void register_task<ELEMENTBINARY_BWD_TASK_ID>();

OpTaskInvocation init(ElementBinaryAttrs const &);
OpTaskInvocation forward(ElementBinaryAttrs const &);
OpTaskInvocation backward(ElementBinaryAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ElementBinaryAttrs const &attrs,
                                  ParallelTensorShape const &lhs_shape,
                                  ParallelTensorShape const &rhs_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);
} // namespace FlexFlow

#endif

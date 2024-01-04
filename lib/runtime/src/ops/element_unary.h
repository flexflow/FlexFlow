#ifndef _ELEMENT_UNARY_H
#define _ELEMENT_UNARY_H

#include "op-attrs/ops/element_unary.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<ELEMENTUNARY_INIT_TASK_ID>();
template <>
void register_task<ELEMENTUNARY_FWD_TASK_ID>();
template <>
void register_task<ELEMENTUNARY_BWD_TASK_ID>();

OpTaskInvocation init(ElementUnaryAttrs const &);
OpTaskInvocation forward(ElementUnaryAttrs const &);
OpTaskInvocation backward(ElementUnaryAttrs const &);

OpTaskInvocation init(ElementScalarUnaryAttrs const &);
OpTaskInvocation forward(ElementScalarUnaryAttrs const &);
OpTaskInvocation backward(ElementScalarUnaryAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ElementUnaryAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ElementScalarUnaryAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

#ifndef _ELEMENT_UNARY_H
#define _ELEMENT_UNARY_H

#include "op-attrs/ops/element_unary.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

using ElementUnaryUnifiedAttrs =
    variant<ElementUnaryAttrs, ElementScalarUnaryAttrs>;

template <>
void register_task<ELEMENTUNARY_INIT_TASK_ID>();
template <>
void register_task<ELEMENTUNARY_FWD_TASK_ID>();
template <>
void register_task<ELEMENTUNARY_BWD_TASK_ID>();

OpTaskInvocation init(ElementUnaryUnifiedAttrs const &);
OpTaskInvocation forward(ElementUnaryUnifiedAttrs const &);
OpTaskInvocation backward(ElementUnaryUnifiedAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ElementUnaryUnifiedAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

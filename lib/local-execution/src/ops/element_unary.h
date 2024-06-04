#ifndef _ELEMENT_UNARY_H
#define _ELEMENT_UNARY_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/element_unary.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(ElementUnaryUnifiedAttrs const &);

TaskImplFunction get_element_unary_init_task_impl();
TaskImplFunction get_element_unary_fwd_task_impl();
TaskImplFunction get_element_unary_bwd_task_impl();

OpTaskSignature get_element_unary_init_signature();
OpTaskSignature get_element_unary_fwd_signature();
OpTaskSignature get_element_unary_bwd_signature();

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

#ifndef _FLEXFLOW_ELEMENT_BINARY_H
#define _FLEXFLOW_ELEMENT_BINARY_H

#include "local-execution/sim_environment.h"
#include "local-execution/task_signature_impl.h"
#include "op-attrs/ops/element_binary_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(ElementBinaryAttrs const &);

OpTaskInvocation init(ElementBinaryAttrs const &);
OpTaskInvocation forward(ElementBinaryAttrs const &);
OpTaskInvocation backward(ElementBinaryAttrs const &);

TaskImplFunction get_element_binary_init_task_impl();
TaskImplFunction get_element_binary_fwd_task_impl();
TaskImplFunction get_element_binary_bwd_task_impl();

OpTaskSignature get_element_binary_init_signature();
OpTaskSignature get_element_binary_fwd_signature();
OpTaskSignature get_element_binary_bwd_signature();

} // namespace FlexFlow

#endif

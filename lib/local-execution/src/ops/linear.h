#ifndef _FLEXFLOW_LINEAR_H
#define _FLEXFLOW_LINEAR_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/linear_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(LinearAttrs const &);

OpTaskInvocation init(LinearAttrs const &);
OpTaskInvocation forward(LinearAttrs const &);
OpTaskInvocation backward(LinearAttrs const &);

TaskImplFunction get_linear_init_task_impl();
TaskImplFunction get_linear_fwd_task_impl();
TaskImplFunction get_linear_bwd_task_impl();

OpTaskSignature get_linear_init_signature();
OpTaskSignature get_linear_fwd_signature();
OpTaskSignature get_linear_bwd_signature();

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_CONCAT_H
#define _FLEXFLOW_CONCAT_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/concat_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(ConcatAttrs const &);

TaskImplFunction get_concat_fwd_task_impl();
TaskImplFunction get_concat_bwd_task_impl();

OpTaskSignature get_concat_fwd_signature();
OpTaskSignature get_concat_bwd_signature();

OpTaskInvocation forward(ConcatAttrs const &);
OpTaskInvocation backward(ConcatAttrs const &);

} // namespace FlexFlow

#endif

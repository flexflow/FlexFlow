#ifndef _FLEXFLOW_DROPOUT_H
#define _FLEXFLOW_DROPOUT_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "local-execution/task_id_t.dtg.h"
#include "op-attrs/ops/dropout_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(DropoutAttrs const &);

TaskImplFunction get_dropout_init_task_impl();
TaskImplFunction get_dropout_fwd_task_impl();
TaskImplFunction get_dropout_bwd_task_impl();

OpTaskSignature get_dropout_init_signature();
OpTaskSignature get_dropout_fwd_signature();
OpTaskSignature get_dropout_bwd_signature();

OpTaskInvocation init(DropoutAttrs const &);
OpTaskInvocation forward(DropoutAttrs const &);
OpTaskInvocation backward(DropoutAttrs const &);

} // namespace FlexFlow

#endif

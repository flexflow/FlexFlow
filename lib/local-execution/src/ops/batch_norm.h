#ifndef _FLEXFLOW_BATCH_NORM_H
#define _FLEXFLOW_BATCH_NORM_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(BatchNormAttrs const &);

TaskImplFunction get_batch_norm_init_task_impl();
TaskImplFunction get_batch_norm_fwd_task_impl();
TaskImplFunction get_batch_norm_bwd_task_impl();

OpTaskSignature get_batch_norm_init_signature();
OpTaskSignature get_batch_norm_fwd_signature();
OpTaskSignature get_batch_norm_bwd_signature();

OpTaskInvocation init(BatchNormAttrs const &);
OpTaskInvocation forward(BatchNormAttrs const &);
OpTaskInvocation backward(BatchNormAttrs const &);

} // namespace FlexFlow

#endif

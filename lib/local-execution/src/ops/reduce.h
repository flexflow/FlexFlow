#ifndef _FLEXFLOW_RUNTIME_SRC_OPS_REDUCE_H
#define _FLEXFLOW_RUNTIME_SRC_OPS_REDUCE_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/reduce_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(ReduceAttrs const &);

TaskImplFunction get_reduce_init_task_impl();
TaskImplFunction get_reduce_fwd_task_impl();
TaskImplFunction get_reduce_bwd_task_impl();

OpTaskSignature get_reduce_init_signature();
OpTaskSignature get_reduce_fwd_signature();
OpTaskSignature get_reduce_bwd_signature();

OpTaskInvocation init(ReduceAttrs const &);
OpTaskInvocation forward(ReduceAttrs const &);
OpTaskInvocation backward(ReduceAttrs const &);

} // namespace FlexFlow

#endif

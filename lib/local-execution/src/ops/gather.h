#ifndef _FLEXFLOW_GATHER_H
#define _FLEXFLOW_GATHER_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/gather_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(GatherAttrs const &);

TaskImplFunction get_gather_init_task_impl();
TaskImplFunction get_gather_fwd_task_impl();
TaskImplFunction get_gather_bwd_task_impl();

OpTaskSignature get_gather_init_signature();
OpTaskSignature get_gather_fwd_signature();
OpTaskSignature get_gather_bwd_signature();

OpTaskInvocation init(GatherAttrs const &);
OpTaskInvocation forward(GatherAttrs const &);
OpTaskInvocation backward(GatherAttrs const &);

} // namespace FlexFlow

#endif

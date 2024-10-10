#ifndef _FLEXFLOW_FLAT_H
#define _FLEXFLOW_FLAT_H

#include "local-execution/sim_environment.h"
#include "op-attrs/ops/flat_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(FlatAttrs const &);

TaskImplFunction get_flat_fwd_task_impl();
TaskImplFunction get_flat_bwd_task_impl();

OpTaskSignature get_flat_fwd_signature();
OpTaskSignature get_flat_bwd_signature();

OpTaskInvocation forward(FlatAttrs const &);
OpTaskInvocation backward(FlatAttrs const &);

} // namespace FlexFlow

#endif

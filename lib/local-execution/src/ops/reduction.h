#ifndef _FLEXFLOW_REDUCTION_H
#define _FLEXFLOW_REDUCTION_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/reduction_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(ReductionAttrs const &);

TaskImplFunction get_reduction_fwd_task_impl();
TaskImplFunction get_reduction_bwd_task_impl();

OpTaskSignature get_reduction_fwd_signature();
OpTaskSignature get_reduction_bwd_signature();

OpTaskInvocation init(ReductionAttrs const &);
OpTaskInvocation forward(ReductionAttrs const &);
OpTaskInvocation backward(ReductionAttrs const &);

} // namespace FlexFlow

#endif

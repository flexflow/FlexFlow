#ifndef _FLEXFLOW_SOFTMAX_H
#define _FLEXFLOW_SOFTMAX_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/softmax_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(SoftmaxAttrs const &);

TaskImplFunction get_softmax_init_task_impl();
TaskImplFunction get_softmax_fwd_task_impl();
TaskImplFunction get_softmax_bwd_task_impl();

OpTaskSignature get_softmax_init_signature();
OpTaskSignature get_softmax_fwd_signature();
OpTaskSignature get_softmax_bwd_signature();

OpTaskInvocation init(SoftmaxAttrs const &);
OpTaskInvocation forward(SoftmaxAttrs const &);
OpTaskInvocation backward(SoftmaxAttrs const &);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_PARTITION_H
#define _FLEXFLOW_PARTITION_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/repartition_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(RepartitionAttrs const &);

TaskImplFunction get_repartition_init_task_impl();
TaskImplFunction get_repartition_fwd_task_impl();
TaskImplFunction get_repartition_bwd_task_impl();

OpTaskSignature get_repartition_init_signature();
OpTaskSignature get_repartition_fwd_signature();
OpTaskSignature get_repartition_bwd_signature();

OpTaskInvocation init(RepartitionAttrs const &);
OpTaskInvocation forward(RepartitionAttrs const &);
OpTaskInvocation backward(RepartitionAttrs const &);

} // namespace FlexFlow

#endif

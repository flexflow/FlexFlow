#ifndef _FLEXFLOW_REPLICATE_H
#define _FLEXFLOW_REPLICATE_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/replicate.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(ReplicateAttrs const &);

TaskImplFunction get_replicate_fwd_task_impl();
TaskImplFunction get_replicate_bwd_task_impl();

OpTaskSignature get_replicate_fwd_signature();
OpTaskSignature get_replicate_bwd_signature();

OpTaskInvocation forward(ReplicateAttrs const &);
OpTaskInvocation backward(ReplicateAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReplicateAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);
} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_PARTITION_H
#define _FLEXFLOW_PARTITION_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/repartition.h"

namespace FlexFlow {

TaskImplFunction get_repartition_init_task_impl();
TaskImplFunction get_repartition_fwd_task_impl();
TaskImplFunction get_repartition_bwd_task_impl();

OpTaskSignature get_repartition_init_signature();
OpTaskSignature get_repartition_fwd_signature();
OpTaskSignature get_repartition_bwd_signature();

OpTaskInvocation init(RepartitionAttrs const &);
OpTaskInvocation forward(RepartitionAttrs const &);
OpTaskInvocation backward(RepartitionAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  RepartitionAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

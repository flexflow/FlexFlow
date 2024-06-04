#ifndef _FLEXFLOW_TOPK_H_
#define _FLEXFLOW_TOPK_H_

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/topk.h"

namespace FlexFlow {

TaskImplFunction get_topk_init_task_impl();
TaskImplFunction get_topk_fwd_task_impl();
TaskImplFunction get_topk_bwd_task_impl();

OpTaskSignature get_topk_init_signature();
OpTaskSignature get_topk_fwd_signature();
OpTaskSignature get_topk_bwd_signature();

OpTaskInvocation init(TopKAttrs const &);
OpTaskInvocation forward(TopKAttrs const &);
OpTaskInvocation backward(TopKAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  TopKAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

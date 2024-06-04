#ifndef _FLEXFLOW_RUNTIME_SRC_OPS_LAYER_NORM_H
#define _FLEXFLOW_RUNTIME_SRC_OPS_LAYER_NORM_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/layer_norm.h"

namespace FlexFlow {

TaskImplFunction get_layer_norm_init_task_impl();
TaskImplFunction get_layer_norm_fwd_task_impl();
TaskImplFunction get_layer_norm_bwd_task_impl();

OpTaskSignature get_layer_norm_init_signature();
OpTaskSignature get_layer_norm_fwd_signature();
OpTaskSignature get_layer_norm_bwd_signature();

OpTaskInvocation init(LayerNormAttrs const &);
OpTaskInvocation forward(LayerNormAttrs const &);
OpTaskInvocation backward(LayerNormAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  LayerNormAttrs const &,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_ATTENTION_H
#define _FLEXFLOW_ATTENTION_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/attention.h"

namespace FlexFlow {

TaskImplFunction get_attention_init_task_impl();
TaskImplFunction get_attention_fwd_task_impl();
TaskImplFunction get_attention_bwd_task_impl();

OpTaskSignature get_attention_init_signature();
OpTaskSignature get_attention_fwd_signature();
OpTaskSignature get_attention_bwd_signature();

OpTaskInvocation init(MultiHeadAttentionAttrs const &);
OpTaskInvocation forward(MultiHeadAttentionAttrs const &);
OpTaskInvocation backward(MultiHeadAttentionAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  MultiHeadAttentionAttrs const &attrs,
                                  InputParallelTensorDesc const &query_shape,
                                  InputParallelTensorDesc const &key_shape,
                                  InputParallelTensorDesc const &value_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv);
} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_EMBEDDING_H
#define _FLEXFLOW_EMBEDDING_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/embedding.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(EmbeddingAttrs const &);

TaskImplFunction get_embedding_fwd_task_impl();
TaskImplFunction get_embedding_bwd_task_impl();

OpTaskSignature get_embedding_fwd_signature();
OpTaskSignature get_embedding_bwd_signature();

OpTaskInvocation forward(EmbeddingAttrs const &);
OpTaskInvocation backward(EmbeddingAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  EmbeddingAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif

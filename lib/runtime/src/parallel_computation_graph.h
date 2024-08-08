#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H

#include "legion_parallel_tensor_shape.h"
#include "op-attrs/operator_attrs.h"
#include "pcg/operator_guid_t.h"
#include "pcg/optimizer.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_tensor.h"
#include "task_spec/op_task_invocation.h"
#include "utils/graph.h"
#include "utils/strong_typedef.h"
#include <type_traits>

namespace FlexFlow {

OpTaskInvocation forward(PCGOperatorAttrs const &);
OpTaskInvocation backward(PCGOperatorAttrs const &);

OpTaskInvocation forward(ParallelComputationGraph const &,
                         operator_guid_t const &);
OpTaskInvocation backward(ParallelComputationGraph const &,
                          operator_guid_t const &);

std::unordered_map<operator_guid_t, OpTaskInvocation>
    init(ParallelComputationGraph const &);
std::unordered_map<operator_guid_t, OpTaskInvocation>
    forward(ParallelComputationGraph const &);
std::unordered_map<operator_guid_t, OpTaskInvocation>
    backward(ParallelComputationGraph const &);
std::unordered_map<operator_guid_t, OpTaskInvocation>
    update(ParallelComputationGraph const &, Optimizer const &);

IndexTaskArgSpec resolve(ParallelComputationGraph const &,
                         operator_guid_t const &,
                         OpArgRefSpec const &);
IndexTaskArgSpec resolve(ParallelComputationGraph const &,
                         operator_guid_t const &,
                         OpArgSpec const &);
parallel_tensor_guid_t resolve(ParallelComputationGraph const &,
                               operator_guid_t const &,
                               OpTensorSpec const &,
                               IsGrad const &);
TaskBinding resolve(ParallelComputationGraph const &,
                    operator_guid_t const &,
                    OpTaskBinding const &);

TaskInvocation resolve(ParallelComputationGraph const &,
                       operator_guid_t const &,
                       OpTaskInvocation const &);
std::unordered_map<operator_guid_t, TaskInvocation>
    resolve(ParallelComputationGraph const &,
            std::unordered_map<operator_guid_t, OpTaskInvocation> const &);

std::unordered_map<std::size_t, parallel_tensor_guid_t>
    get_input_ptensors_by_idx(ParallelComputationGraph const &,
                              operator_guid_t);
std::unordered_map<std::size_t, parallel_tensor_guid_t>
    get_weight_ptensors_by_idx(ParallelComputationGraph const &,
                               operator_guid_t);
std::unordered_map<std::size_t, parallel_tensor_guid_t>
    get_output_ptensors_by_idx(ParallelComputationGraph const &,
                               operator_guid_t);

static_assert(std::is_copy_constructible<ParallelComputationGraph>::value, "");
static_assert(std::is_move_constructible<ParallelComputationGraph>::value, "");
static_assert(std::is_copy_assignable<ParallelComputationGraph>::value, "");
static_assert(std::is_move_assignable<ParallelComputationGraph>::value, "");

} // namespace FlexFlow

#endif

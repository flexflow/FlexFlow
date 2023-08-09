#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H

#include "legion_parallel_tensor_shape.h"
#include "op_task_invocation.h"
#include "operator.h"
#include "operator_guid_t.h"
#include "optimizer.h"
#include "parallel_tensor.h"
#include "utils/graph.h"
#include "utils/strong_typedef.h"
#include <type_traits>

namespace FlexFlow {

class ParallelComputationGraph {
public:
  ParallelComputationGraph() = delete;
  ParallelComputationGraph(
      LabelledOpenMultiDiGraph<Operator, ParallelTensor> const &);

  Operator const &at(operator_guid_t const &) const;
  Operator &at(operator_guid_t);

  Operator const &operator[](operator_guid_t const &) const;
  Operator &operator[](operator_guid_t);

  ParallelTensor const &at(parallel_tensor_guid_t const &) const;
  ParallelTensor &at(parallel_tensor_guid_t);

  ParallelTensor const &operator[](parallel_tensor_guid_t const &) const;
  ParallelTensor &operator[](parallel_tensor_guid_t);

  template <typename F>
  ParallelComputationGraph on_underlying(F const &f) {
    using Underlying = LabelledOpenMultiDiGraph<Operator, ParallelTensor>;
    static_assert(std::is_same<decltype(std::declval<F>()(
                                   std::declval<Underlying const &>())),
                               Underlying>::value,
                  "Function must return an value of the type underlying PCG");
  }

  friend void swap(ParallelComputationGraph &, ParallelComputationGraph &);

public:
  LabelledOpenMultiDiGraph<Operator, ParallelTensor> graph;
};

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

ArgSpec resolve(ParallelComputationGraph const &,
                operator_guid_t const &,
                OpArgRefSpec const &);
ArgSpec resolve(ParallelComputationGraph const &,
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

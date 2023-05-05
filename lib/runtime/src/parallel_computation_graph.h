#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H

#include "parallel_tensor.h"
#include "operator.h"
#include "utils/graph.h"
#include "legion_parallel_tensor_shape.h"
#include "utils/strong_typedef.h"
#include "operator_guid_t.h"
#include "op_task_spec.h"

namespace FlexFlow {

class ParallelComputationGraph {
public:
  ParallelComputationGraph() = delete;
  
  Operator const &at(operator_guid_t) const;
  Operator &at(operator_guid_t);

  Operator const &operator[](operator_guid_t) const;
  Operator &operator[](operator_guid_t);

  ParallelTensor const &at(parallel_tensor_guid_t) const;
  ParallelTensor &at(parallel_tensor_guid_t);

  ParallelTensor const &operator[](parallel_tensor_guid_t) const;
  ParallelTensor &operator[](parallel_tensor_guid_t);

  friend void swap(ParallelComputationGraph &, ParallelComputationGraph &);
private:
  LabelledOpenMultiDiGraph<Operator, ParallelTensor> graph;
};

OpTaskInvocation forward(PCGOperatorAttrs const &);
OpTaskInvocation backward(PCGOperatorAttrs const &);

OpTaskInvocation forward(ParallelComputationGraph const &, operator_guid_t);
OpTaskInvocation backward(ParallelComputationGraph const &, operator_guid_t);

TaskInvocation resolve_op_task_spec(ParallelComputationGraph const &, OpTaskInvocation const &);


static_assert(std::is_copy_constructible<ParallelComputationGraph>::value, "");
static_assert(std::is_move_constructible<ParallelComputationGraph>::value, "");
static_assert(std::is_copy_assignable<ParallelComputationGraph>::value, "");
static_assert(std::is_move_assignable<ParallelComputationGraph>::value, "");

}

#endif

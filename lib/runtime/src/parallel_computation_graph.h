#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H

#include "parallel_tensor.h"
#include "operator.h"
#include "utils/graph.h"
#include "legion_parallel_tensor_shape.h"
#include "utils/strong_typedef.h"
#include "operator_guid_t.h"

namespace FlexFlow {

class ParallelComputationGraph {
public:
  ParallelComputationGraph() = delete;

  optional<Operator> get_source(ParallelTensor const &) const;

  std::vector<Operator> get_operators() const;
  Operator get_final_operator() const;
private:
  ParallelTensor create_parallel_tensor(ParallelTensorShape const &,
                                        CreateGrad create_grad = CreateGrad::YES);
  ParallelTensor create_parallel_tensor(LegionParallelTensorShape const &,
                                        CreateGrad create_grad = CreateGrad::YES);
  ParallelParameter create_parallel_weight(ParallelTensorShape const &,
                                           CreateGrad create_grad = CreateGrad::YES,
                                           Initializer *initializer = nullptr,
                                           ParameterSyncType sync_type = ParameterSyncType::NONE);
  ParallelParameter create_parallel_weight(LegionParallelTensorShape const &,
                                           CreateGrad create_grad = CreateGrad::YES,
                                           Initializer *initializer = nullptr,
                                           ParameterSyncType sync_type = ParameterSyncType::NONE);
  
  friend void swap(ParallelComputationGraph &, ParallelComputationGraph &);
private:
  LabelledOpenMultiDiGraph<Operator, ParallelTensor> graph;
};

static_assert(std::is_copy_constructible<ParallelComputationGraph>::value, "");
static_assert(std::is_move_constructible<ParallelComputationGraph>::value, "");
static_assert(std::is_copy_assignable<ParallelComputationGraph>::value, "");
static_assert(std::is_copy_constructible<ParallelComputationGraph>::value, "");

}

#endif

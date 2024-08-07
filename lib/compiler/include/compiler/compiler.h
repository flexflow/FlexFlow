#ifndef _FLEXFLOW_COMPILER_COMPILER_H
#define _FLEXFLOW_COMPILER_COMPILER_H

#include "pcg/cost_values.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/tensor_mapping.h"

namespace FlexFlow {

enum class SearchAlgorithm {
  DATA_PARALLEL,
};

using SearchAlgorithmConfig = std::variant<>;
using SearchSolution = std::variant<>;

struct SearchResult {
  ParallelComputationGraph pcg;
  TensorMapping tensor_mapping;
  SearchSolution solution;
  CostValues cost_values;
};

SearchResult optimize(ComputationGraph const &,
                      MachineSpecification const &,
                      CostEstimator const &,
                      SearchAlgorithm,
                      optional<AlgorithmConfig> const &);

// struct SearchSolution {
//   LabelledMultiDiGraph<PCGOperatorAttrs, ParallelTensorShape> optimized_pcg;
//   std::unordered_map<Node, MachineView> device_assignments;
//   /* std::unordered_map<tensor_guid_t,
//   std::unordered_set<parallel_tensor_guid_t>> tensor_mappings; */
// };
//
// SearchSolution run_data_parallelize(ComputationGraph const &,
// MachineSpecification const &);

} // namespace FlexFlow

#endif

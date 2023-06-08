#ifndef _FLEXFLOW_COMPILER_COMPILER_H
#define _FLEXFLOW_COMPILER_COMPILER_H

#include "pcg/cost_values.h"
#include "pcg/machine_view.h"

namespace FlexFlow {

enum class SearchAlgorithm {
  DATA_PARALLEL,
};

using SearchAlgorithmConfig = variant<>;
using SearchSolution = variant<>;

struct SearchResult {
  ParallelComputationGraph pcg;
  TensorMapping tensor_mapping;
  SearchSolution solution;
  CostValues cost_values;
};

SearchResult optimize(ComputationGraph const &,
                      MachineSpecification const &,
                      CostEstimator const &,
                      Algorithm,
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

#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_PROBLEM_TREE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_MAPPING_PROBLEM_TREE_H

#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_series_split.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/mm_problem_tree_parallel_split.dtg.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"

namespace FlexFlow {

MachineMappingProblemTree
  mm_problem_tree_make_series_split(AbstractedTensorSetMovement const &tensor_set_movement,
                                    MachineMappingProblemTree const &pre,
                                    MachineMappingProblemTree const &post);
MachineMappingProblemTree
  mm_problem_tree_make_parallel_split(MachineMappingProblemTree const &lhs,
                                      MachineMappingProblemTree const &rhs);
MachineMappingProblemTree mm_problem_tree_make_leaf(PCGOperatorAttrs const &);

SPDecompositionTreeNodeType get_node_type(MachineMappingProblemTree const &);

MMProblemTreeSeriesSplit require_series_split(MachineMappingProblemTree const &);
MMProblemTreeParallelSplit require_parallel_split(MachineMappingProblemTree const &);
PCGOperatorAttrs require_leaf(MachineMappingProblemTree const &);

std::unordered_multiset<PCGOperatorAttrs> get_leaves(MachineMappingProblemTree const &);

template <typename Result, typename F>
Result visit(MachineMappingProblemTree const &t, F &&f) {
  SPDecompositionTreeNodeType node_type = get_node_type(t);
  switch (node_type) {
    case SPDecompositionTreeNodeType::SERIES: {
      Result result = f(require_series_split(t));
      return result;
    }
    case SPDecompositionTreeNodeType::PARALLEL: {
      Result result = f(require_parallel_split(t));
      return result;
    }
    case SPDecompositionTreeNodeType::NODE: {
      Result result = f(require_leaf(t));
      return result;
    }
    default:
      throw mk_runtime_error(fmt::format("Unknown SPDecompositionTreeNodeType: {}", node_type));
  }
}

} // namespace FlexFlow

#endif

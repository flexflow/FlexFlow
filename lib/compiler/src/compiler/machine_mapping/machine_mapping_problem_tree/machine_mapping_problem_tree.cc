#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_all_leaf_paths.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_subtree_at_path.h"

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<MachineMappingProblemTree,
                                               MMProblemTreeSeriesSplit,
                                               MMProblemTreeParallelSplit,
                                               UnmappedOpCostEstimateKey>
    generic_binary_sp_impl_for_mm_problem_tree() {
  return GenericBinarySPDecompositionTreeImplementation<
      MachineMappingProblemTree,
      MMProblemTreeSeriesSplit,
      MMProblemTreeParallelSplit,
      UnmappedOpCostEstimateKey>{
      /*series_get_left_child=*/[](MMProblemTreeSeriesSplit const &split)
                                    -> MachineMappingProblemTree const & {
        return split.get_left_child();
      },
      /*parallel_get_left_child=*/
      [](MMProblemTreeParallelSplit const &split)
          -> MachineMappingProblemTree const & {
        return split.get_left_child();
      },
      /*series_get_right_child=*/
      [](MMProblemTreeSeriesSplit const &split)
          -> MachineMappingProblemTree const & {
        return split.get_right_child();
      },
      /*parallel_get_right_child=*/
      [](MMProblemTreeParallelSplit const &split)
          -> MachineMappingProblemTree const & {
        return split.get_right_child();
      },
      /*get_node_type=*/
      [](MachineMappingProblemTree const &tree) -> SPDecompositionTreeNodeType {
        return get_node_type(tree);
      },
      /*require_series=*/
      [](MachineMappingProblemTree const &tree)
          -> MMProblemTreeSeriesSplit const & {
        return tree.get<MMProblemTreeSeriesSplit>();
      },
      /*require_parallel=*/
      [](MachineMappingProblemTree const &tree)
          -> MMProblemTreeParallelSplit const & {
        return tree.get<MMProblemTreeParallelSplit>();
      },
      /*require_leaf=*/
      [](MachineMappingProblemTree const &tree)
          -> UnmappedOpCostEstimateKey const & {
        return tree.get<UnmappedOpCostEstimateKey>();
      },
  };
}

SPDecompositionTreeNodeType
    get_node_type(MachineMappingProblemTree const &tree) {
  return tree.visit<SPDecompositionTreeNodeType>(overload{
      [](MMProblemTreeSeriesSplit const &) {
        return SPDecompositionTreeNodeType::SERIES;
      },
      [](MMProblemTreeParallelSplit const &) {
        return SPDecompositionTreeNodeType::PARALLEL;
      },
      [](UnmappedOpCostEstimateKey const &) {
        return SPDecompositionTreeNodeType::NODE;
      },
  });
}

std::unordered_multiset<UnmappedOpCostEstimateKey>
    get_leaves(MachineMappingProblemTree const &tree) {
  return get_leaves(tree, generic_binary_sp_impl_for_mm_problem_tree());
}

std::unordered_set<BinaryTreePath>
    get_all_leaf_paths(MachineMappingProblemTree const &tree) {
  return get_all_leaf_paths(tree, generic_binary_sp_impl_for_mm_problem_tree());
}

std::optional<MachineMappingProblemTree>
    mm_problem_tree_get_subtree_at_path(MachineMappingProblemTree const &tree,
                                        BinaryTreePath const &path) {
  return get_subtree_at_path(
      tree, generic_binary_sp_impl_for_mm_problem_tree(), path);
}

} // namespace FlexFlow

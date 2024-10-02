#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_subtree_at_path.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_node_type.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/require.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_all_leaf_paths.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/wrap.h"

namespace FlexFlow {

MachineMappingProblemTree mm_problem_tree_make_series_split(AbstractedTensorSetMovement const &tensor_set_movement,
                                                            MachineMappingProblemTree const &lhs, 
                                                            MachineMappingProblemTree const &rhs) {
  return MachineMappingProblemTree{
    make_generic_binary_series_split(
      MMProblemTreeSeriesSplitLabel{tensor_set_movement},
      lhs.raw_tree,
      rhs.raw_tree),
  };
}

MachineMappingProblemTree mm_problem_tree_make_parallel_split(MachineMappingProblemTree const &lhs,
                                                              MachineMappingProblemTree const &rhs) {
  return MachineMappingProblemTree{
    make_generic_binary_parallel_split(
      MMProblemTreeParallelSplitLabel{},
      lhs.raw_tree,
      rhs.raw_tree),
  };
}

MachineMappingProblemTree mm_problem_tree_make_leaf(UnmappedOpCostEstimateKey const &leaf_label) {
  return MachineMappingProblemTree{
    make_generic_binary_sp_leaf<
      MMProblemTreeSeriesSplitLabel,
      MMProblemTreeParallelSplitLabel,
      UnmappedOpCostEstimateKey>(leaf_label),
  };
}

SPDecompositionTreeNodeType get_node_type(MachineMappingProblemTree const &tree) {
  return get_node_type(tree.raw_tree);
}


MMProblemTreeSeriesSplit require_series_split(MachineMappingProblemTree const &t) {
  return MMProblemTreeSeriesSplit{
    require_series(t.raw_tree),
  };
}

MMProblemTreeParallelSplit require_parallel_split(MachineMappingProblemTree const &t) {
  return MMProblemTreeParallelSplit{
    require_parallel(t.raw_tree),
  };
}

UnmappedOpCostEstimateKey require_leaf(MachineMappingProblemTree const &t) {
  return require_leaf(t.raw_tree);
}

MachineMappingProblemTree wrap_series_split(MMProblemTreeSeriesSplit const &series) {
  return MachineMappingProblemTree{
    wrap_series_split(series.raw_split),
  };
}

MachineMappingProblemTree wrap_parallel_split(MMProblemTreeParallelSplit const &parallel) {
  return MachineMappingProblemTree{
    wrap_parallel_split(parallel.raw_split),
  };
}

std::unordered_multiset<UnmappedOpCostEstimateKey> get_leaves(MachineMappingProblemTree const &t) {
  return get_leaves(t.raw_tree);
}

std::unordered_set<BinaryTreePath> get_all_leaf_paths(MachineMappingProblemTree const &t) {
  return get_all_leaf_paths(t.raw_tree);
}

std::optional<MachineMappingProblemTree> mm_problem_tree_get_subtree_at_path(MachineMappingProblemTree const &tree,
                                                                             BinaryTreePath const &path) {
  std::optional<GenericBinarySPDecompositionTree<
    MMProblemTreeSeriesSplitLabel, MMProblemTreeParallelSplitLabel, UnmappedOpCostEstimateKey
  >> raw_subtree = get_subtree_at_path(tree.raw_tree, path);

  if (!raw_subtree.has_value()) {
    return std::nullopt;
  } else {
    return MachineMappingProblemTree{raw_subtree.value()};
  }
}

} // namespace FlexFlow

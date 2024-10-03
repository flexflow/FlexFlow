#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/make.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/require.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/get_node_type.h"

namespace FlexFlow {

BinarySPDecompositionTree
    make_series_split(BinarySPDecompositionTree const &lhs,
                      BinarySPDecompositionTree const &rhs) {
  return BinarySPDecompositionTree{
      leaf_only_make_series_split(lhs.raw_tree, rhs.raw_tree),
  };
}

BinarySPDecompositionTree
    make_parallel_split(BinarySPDecompositionTree const &lhs,
                        BinarySPDecompositionTree const &rhs) {
  return BinarySPDecompositionTree{
      leaf_only_make_parallel_split(lhs.raw_tree, rhs.raw_tree),
  };
}

BinarySPDecompositionTree make_leaf_node(Node const &n) {
  return BinarySPDecompositionTree{
      leaf_only_make_leaf_node(n),
  };
}

bool is_binary_sp_tree_left_associative(BinarySPDecompositionTree const &tt) {
  return is_binary_sp_tree_left_associative(tt.raw_tree);
}

bool is_binary_sp_tree_right_associative(BinarySPDecompositionTree const &tt) {
  return is_binary_sp_tree_right_associative(tt.raw_tree);
}

std::unordered_multiset<Node> get_leaves(BinarySPDecompositionTree const &tt) {
  return get_leaves(tt.raw_tree);
}

BinarySeriesSplit require_series(BinarySPDecompositionTree const &tt) {
  return BinarySeriesSplit{
    require_leaf_only_binary_series_split(tt.raw_tree),
  };
}

BinaryParallelSplit require_parallel(BinarySPDecompositionTree const &tt) {
  return BinaryParallelSplit{
    require_leaf_only_binary_parallel_split(tt.raw_tree),
  };
}

Node require_leaf(BinarySPDecompositionTree const &tt) {
  return require_leaf_only_binary_leaf(tt.raw_tree);
}

SPDecompositionTreeNodeType get_node_type(BinarySPDecompositionTree const &tt) {
  return get_node_type(tt.raw_tree);
}


} // namespace FlexFlow

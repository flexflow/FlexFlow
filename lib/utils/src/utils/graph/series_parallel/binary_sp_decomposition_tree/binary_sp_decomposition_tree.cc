#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"

namespace FlexFlow {

GenericBinarySPDecompositionTreeImplementation<
  BinarySPDecompositionTree,
  BinarySeriesSplit,
  BinaryParallelSplit,
  Node> generic_impl_for_binary_sp_tree() {
  NOT_IMPLEMENTED();
}

bool is_binary_sp_tree_left_associative(BinarySPDecompositionTree const &tree) {
  return is_binary_sp_tree_left_associative(tree, generic_impl_for_binary_sp_tree());
}

bool is_binary_sp_tree_right_associative(BinarySPDecompositionTree const &tree) {
  return is_binary_sp_tree_right_associative(tree, generic_impl_for_binary_sp_tree());
}

std::unordered_multiset<Node> get_leaves(BinarySPDecompositionTree const &tree) {
  return get_leaves(tree, generic_impl_for_binary_sp_tree());
}

} // namespace FlexFlow

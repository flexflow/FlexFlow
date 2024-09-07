#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_nodes.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"

namespace FlexFlow {

BinarySPDecompositionTree
    make_series_split(BinarySPDecompositionTree const &lhs,
                      BinarySPDecompositionTree const &rhs) {
  return BinarySPDecompositionTree{
      make_generic_binary_series_split(lhs.raw_tree, rhs.raw_tree),
  };
}

BinarySPDecompositionTree
    make_parallel_split(BinarySPDecompositionTree const &lhs,
                        BinarySPDecompositionTree const &rhs) {
  return BinarySPDecompositionTree{
      make_generic_binary_parallel_split(lhs.raw_tree, rhs.raw_tree),
  };
}

BinarySPDecompositionTree make_leaf_node(Node const &n) {
  return BinarySPDecompositionTree{
      make_generic_binary_sp_leaf(n),
  };
}

bool is_binary_sp_tree_left_associative(BinarySPDecompositionTree const &tt) {
  return is_binary_sp_tree_left_associative(tt.raw_tree);
}

bool is_binary_sp_tree_right_associative(BinarySPDecompositionTree const &tt) {
  return is_binary_sp_tree_right_associative(tt.raw_tree);
}

std::unordered_multiset<Node> get_nodes(BinarySPDecompositionTree const &tt) {
  return get_nodes(tt.raw_tree);
}

} // namespace FlexFlow

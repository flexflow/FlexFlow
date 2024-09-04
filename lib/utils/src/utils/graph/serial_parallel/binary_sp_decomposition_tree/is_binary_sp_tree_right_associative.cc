#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/overload.h"

namespace FlexFlow {

bool is_binary_sp_tree_right_associative(BinarySPDecompositionTree const &t) {
  return t.visit<bool>(overload {
    [](Node const &n) { return true; },
    [](BinarySeriesSplit const &s) { 
      return !s.left_child().has<BinarySeriesSplit>()
        && is_binary_sp_tree_right_associative(s.left_child())
        && is_binary_sp_tree_right_associative(s.right_child());
    },
    [](BinaryParallelSplit const &p) {
      return !p.left_child().has<BinaryParallelSplit>()
        && is_binary_sp_tree_right_associative(p.left_child())
        && is_binary_sp_tree_right_associative(p.right_child());
    },
  });
}

} // namespace FlexFlow

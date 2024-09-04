#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/get_num_tree_nodes.h"
#include "utils/overload.h"

namespace FlexFlow {

int get_num_tree_nodes(BinarySPDecompositionTree const &t) {
  return t.visit<int>(overload {
    [](Node const &n) { return 1; },
    [](BinarySeriesSplit const &s) { return get_num_tree_nodes(s); },
    [](BinaryParallelSplit const &p) { return get_num_tree_nodes(p); },
  });
}

int get_num_tree_nodes(BinarySeriesSplit const &s) {
  return 1 + get_num_tree_nodes(s.left_child()) + get_num_tree_nodes(s.right_child());
}

int get_num_tree_nodes(BinaryParallelSplit const &p) {
  return 1 + get_num_tree_nodes(p.left_child()) + get_num_tree_nodes(p.right_child());
}

} // namespace FlexFlow

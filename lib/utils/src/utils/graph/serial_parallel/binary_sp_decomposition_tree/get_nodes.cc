#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/get_nodes.h"
#include "utils/overload.h"
#include "utils/containers/multiset_union.h"

namespace FlexFlow {

std::unordered_multiset<Node> get_nodes(BinarySPDecompositionTree const &t) {
  return t.visit<std::unordered_multiset<Node>>(overload {
    [](Node const &n) { return std::unordered_multiset{n}; },
    [](BinarySeriesSplit const &s) { return get_nodes(s); },
    [](BinaryParallelSplit const &p) { return get_nodes(p); },
  });
}

std::unordered_multiset<Node> get_nodes(BinarySeriesSplit const &s) {
  return multiset_union(get_nodes(s.left_child()), get_nodes(s.right_child()));
}

std::unordered_multiset<Node> get_nodes(BinaryParallelSplit const &p) {
  return multiset_union(get_nodes(p.left_child()), get_nodes(p.right_child()));
}

} // namespace FlexFlow

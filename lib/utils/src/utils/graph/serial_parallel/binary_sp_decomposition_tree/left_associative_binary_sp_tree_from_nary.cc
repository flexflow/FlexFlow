#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/left_associative_binary_sp_tree_from_nary.h"
#include "utils/overload.h"
#include "utils/containers/transform.h"
#include "utils/containers/foldl1.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

BinarySPDecompositionTree left_associative_binary_sp_tree_from_nary(SerialParallelDecomposition const &nary) {
  std::function<BinarySPDecompositionTree(std::variant<ParallelSplit, Node> const &)> from_serial_child;
  std::function<BinarySPDecompositionTree(std::variant<SerialSplit, Node> const &)> from_parallel_child;

  auto from_node = [](Node const &n) {
    return BinarySPDecompositionTree{n};
  };

  auto from_serial = [&](SerialSplit const &s) {
    std::vector<BinarySPDecompositionTree> children = transform(s.children, from_serial_child);
    return foldl1(children, [](BinarySPDecompositionTree const &accum, BinarySPDecompositionTree const &x) {
      return BinarySPDecompositionTree{BinarySeriesSplit{accum, x}};
    });
  };

  auto from_parallel = [&](ParallelSplit const &s) {
    std::vector<BinarySPDecompositionTree> children = transform(vector_of(s.children), from_parallel_child);
    return foldl1(children, [](BinarySPDecompositionTree const &accum, BinarySPDecompositionTree const &x) {
      return BinarySPDecompositionTree{BinaryParallelSplit{accum, x}};
    });
  };

  from_parallel_child = [&](std::variant<SerialSplit, Node> const &v) -> BinarySPDecompositionTree {
    return std::visit(overload {
      [&](Node const &n) { return from_node(n); },
      [&](SerialSplit const &s) { return from_serial(s); },
    }, v);
  };

  from_serial_child = [&](std::variant<ParallelSplit, Node> const &v) -> BinarySPDecompositionTree {
    return std::visit(overload {
      [&](Node const &n) { return from_node(n); },
      [&](ParallelSplit const &p) { return from_parallel(p); },
    }, v);
  };

  return nary.visit<BinarySPDecompositionTree>(overload {
    [&](Node const &n) { return from_node(n); },
    [&](SerialSplit const &s) { return from_serial(s); },
    [&](ParallelSplit const &p) { return from_parallel(p); },
  });
}

} // namespace FlexFlow

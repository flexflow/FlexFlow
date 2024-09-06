#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/left_associative_binary_sp_tree_from_nary.h"
#include "utils/containers/foldl1.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/overload.h"

namespace FlexFlow {

BinarySPDecompositionTree left_associative_binary_sp_tree_from_nary(
    SerialParallelDecomposition const &nary) {
  std::function<GenericBinarySPDecompositionTree<Node>(
      std::variant<ParallelSplit, Node> const &)>
      from_serial_child;
  std::function<GenericBinarySPDecompositionTree<Node>(
      std::variant<SerialSplit, Node> const &)>
      from_parallel_child;

  auto from_node = [](Node const &n) -> GenericBinarySPDecompositionTree<Node> {
    return GenericBinarySPDecompositionTree<Node>{n};
  };

  auto from_serial =
      [&](SerialSplit const &s) -> GenericBinarySPDecompositionTree<Node> {
    std::vector<GenericBinarySPDecompositionTree<Node>> children =
        transform(s.children, from_serial_child);
    return foldl1(children,
                  [](GenericBinarySPDecompositionTree<Node> const &accum,
                     GenericBinarySPDecompositionTree<Node> const &x) {
                    return GenericBinarySPDecompositionTree<Node>{
                        GenericBinarySeriesSplit<Node>{accum, x},
                    };
                  });
  };

  auto from_parallel =
      [&](ParallelSplit const &s) -> GenericBinarySPDecompositionTree<Node> {
    std::vector<GenericBinarySPDecompositionTree<Node>> children =
        transform(vector_of(s.children), from_parallel_child);
    return foldl1(children,
                  [](GenericBinarySPDecompositionTree<Node> const &accum,
                     GenericBinarySPDecompositionTree<Node> const &x) {
                    return GenericBinarySPDecompositionTree<Node>{
                        GenericBinaryParallelSplit<Node>{accum, x}};
                  });
  };

  from_parallel_child = [&](std::variant<SerialSplit, Node> const &v)
      -> GenericBinarySPDecompositionTree<Node> {
    return std::visit(overload{
                          [&](Node const &n) { return from_node(n); },
                          [&](SerialSplit const &s) { return from_serial(s); },
                      },
                      v);
  };

  from_serial_child = [&](std::variant<ParallelSplit, Node> const &v)
      -> GenericBinarySPDecompositionTree<Node> {
    return std::visit(
        overload{
            [&](Node const &n) { return from_node(n); },
            [&](ParallelSplit const &p) { return from_parallel(p); },
        },
        v);
  };

  return BinarySPDecompositionTree{
      nary.visit<GenericBinarySPDecompositionTree<Node>>(overload{
          [&](Node const &n) { return from_node(n); },
          [&](SerialSplit const &s) { return from_serial(s); },
          [&](ParallelSplit const &p) { return from_parallel(p); },
      }),
  };
}

} // namespace FlexFlow

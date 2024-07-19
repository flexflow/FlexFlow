#include "utils/graph/serial_parallel/serialparallel.h"
#include "./serialparallel_internal.h"
#include "./sink_settings.dtg.h"
#include "./source_settings.dtg.h"
#include "utils/containers.h"
#include "utils/graph/algorithms.h"

namespace FlexFlow {

SerialParallelDecomposition
    get_serial_parallel_decomposition(DiGraphView const &g) {
  std::variant<IntermediateSpDecompositionTree, Node> ast = sp_decomposition(g);
  return to_final_ast(ast);
}

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp) {
  return sp.visit<std::unordered_set<Node>>(
      [](auto &&t) { return get_nodes(t); });
}

std::unordered_set<Node> get_nodes(SerialSplit const &serial) {
  return set_union(transform(
      serial.children,
      [](std::variant<ParallelSplit, Node> const &child)
          -> std::unordered_set<Node> {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::unordered_set<Node> get_nodes(ParallelSplit const &parallel) {
  return set_union(transform(
      parallel.children, [](std::variant<SerialSplit, Node> const &child) {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::unordered_set<Node> get_nodes(Node const &node) {
  return {node};
}

} // namespace FlexFlow

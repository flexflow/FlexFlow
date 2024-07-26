#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/graph/serial_parallel/intermediate_sp_decomposition_tree.h"
#include "utils/hash/unordered_set.h"
#include "utils/variant.h"

namespace FlexFlow {

struct ToFinalAST {
  std::variant<SerialSplit, ParallelSplit, Node>
      operator()(IntermediateSpDecompositionTree const &node) {
    if (node.type == SplitType::SERIAL) {
      return SerialSplit{transform(
          node.children,
          [](std::variant<IntermediateSpDecompositionTree, Node> const &s) {
            return narrow<std::variant<ParallelSplit, Node>>(
                       internal_to_final_ast(s))
                .value();
          })};
    } else {
      return ParallelSplit{unordered_set_of(transform(
          node.children,
          [](std::variant<IntermediateSpDecompositionTree, Node> const &s) {
            return narrow<std::variant<SerialSplit, Node>>(
                       internal_to_final_ast(s))
                .value();
          }))};
    }
  }

  std::variant<SerialSplit, ParallelSplit, Node> operator()(Node const &node) {
    return node;
  }
};

std::variant<SerialSplit, ParallelSplit, Node> internal_to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit(ToFinalAST{}, flatten_ast(ast));
}

SerialParallelDecomposition to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit([](auto &&x) { return SerialParallelDecomposition{x}; },
                    internal_to_final_ast(ast));
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

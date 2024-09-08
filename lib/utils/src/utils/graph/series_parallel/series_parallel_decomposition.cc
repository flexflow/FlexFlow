#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/containers/multiset_union.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/series_parallel/intermediate_sp_decomposition_tree.h"
#include "utils/hash/unordered_set.h"
#include "utils/variant.h"

namespace FlexFlow {

struct ToFinalAST {
  std::variant<SeriesSplit, ParallelSplit, Node>
      operator()(IntermediateSpDecompositionTree const &node) {
    if (node.type == SplitType::SERIES) {
      return SeriesSplit{transform(
          node.children,
          [](std::variant<IntermediateSpDecompositionTree, Node> const &s) {
            return narrow<std::variant<ParallelSplit, Node>>(
                       internal_to_final_ast(s))
                .value();
          })};
    } else {
      return ParallelSplit{unordered_multiset_of(transform(
          node.children,
          [](std::variant<IntermediateSpDecompositionTree, Node> const &s) {
            return narrow<std::variant<SeriesSplit, Node>>(
                       internal_to_final_ast(s))
                .value();
          }))};
    }
  }

  std::variant<SeriesSplit, ParallelSplit, Node> operator()(Node const &node) {
    return node;
  }
};

std::variant<SeriesSplit, ParallelSplit, Node> internal_to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit(ToFinalAST{}, flatten_ast(ast));
}

SeriesParallelDecomposition to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit([](auto &&x) { return SeriesParallelDecomposition{x}; },
                    internal_to_final_ast(ast));
}

std::unordered_multiset<Node> get_nodes(SeriesParallelDecomposition const &sp) {
  return sp.visit<std::unordered_multiset<Node>>(
      [](auto &&t) { return get_nodes(t); });
}

std::unordered_multiset<Node> get_nodes(SeriesSplit const &serial) {
  return multiset_union(transform(
      serial.children,
      [](std::variant<ParallelSplit, Node> const &child)
          -> std::unordered_multiset<Node> {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::unordered_multiset<Node> get_nodes(ParallelSplit const &parallel) {
  return multiset_union(transform(
      vector_of(parallel.children),
      [](std::variant<SeriesSplit, Node> const &child) {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::unordered_multiset<Node> get_nodes(Node const &node) {
  return {node};
}

} // namespace FlexFlow

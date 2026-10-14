#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/containers/all_of.h"
#include "utils/containers/extend.h"
#include "utils/containers/get_only.h"
#include "utils/containers/multiset_of.h"
#include "utils/containers/multiset_union.h"
#include "utils/containers/set_union.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/series_parallel/intermediate_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/series_parallel_metrics.h"
#include "utils/hash/set.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/variant.h"
#include <set>

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
      return ParallelSplit{multiset_of(transform(
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

std::multiset<Node> get_nodes(SeriesParallelDecomposition const &sp) {
  return sp.visit<std::multiset<Node>>([](auto &&t) { return get_nodes(t); });
}

std::multiset<Node> get_nodes(SeriesSplit const &serial) {
  return multiset_union(transform(
      serial.children,
      [](std::variant<ParallelSplit, Node> const &child)
          -> std::multiset<Node> {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::multiset<Node> get_nodes(ParallelSplit const &parallel) {
  return multiset_union(transform(
      vector_of(parallel.get_children()),
      [](std::variant<SeriesSplit, Node> const &child) {
        return std::visit([](auto &&t) { return get_nodes(t); }, child);
      }));
}

std::multiset<Node> get_nodes(Node const &node) {
  return {node};
}

bool is_empty(Node const &node) {
  return false;
}

nonnegative_int num_nodes(SeriesParallelDecomposition const &sp) {
  return sum(values(get_num_occurrences_of_nodes(sp)));
}

bool has_no_duplicate_nodes(SeriesParallelDecomposition const &sp) {
  return all_of(values(get_num_occurrences_of_nodes(sp)),
                [](nonnegative_int count) { return count == 1_n; });
}

SeriesParallelDecomposition series_composition(
    std::vector<SeriesParallelDecomposition> const &sp_compositions) {

  ASSERT(sp_compositions.size() > 0,
         "Cannot create series composition with zero elements");

  if (sp_compositions.size() == 1) {
    return get_only(sp_compositions);
  }

  std::vector<std::variant<ParallelSplit, Node>> composition{};
  for (SeriesParallelDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<SeriesSplit>()) {
      extend(composition, sp_comp.get<SeriesSplit>().children);
    } else if (sp_comp.has<ParallelSplit>()) {
      composition.push_back(sp_comp.get<ParallelSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.push_back(sp_comp.get<Node>());
    }
  }
  return SeriesParallelDecomposition{SeriesSplit{composition}};
}

SeriesParallelDecomposition parallel_composition(
    std::multiset<SeriesParallelDecomposition> const &sp_compositions) {

  ASSERT(sp_compositions.size() > 0,
         "Cannot create parallel composition with zero elements");

  if (sp_compositions.size() == 1) {
    return get_only(sp_compositions);
  }

  std::multiset<std::variant<::FlexFlow::SeriesSplit, ::FlexFlow::Node>>
      composition{};
  for (SeriesParallelDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<ParallelSplit>()) {
      composition = multiset_union(
          composition,
          multiset_of(sp_comp.get<ParallelSplit>().get_children()));
    } else if (sp_comp.has<SeriesSplit>()) {
      composition.insert(sp_comp.get<SeriesSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.insert(sp_comp.get<Node>());
    }
  }
  return SeriesParallelDecomposition(ParallelSplit{multiset_of(composition)});
}

} // namespace FlexFlow

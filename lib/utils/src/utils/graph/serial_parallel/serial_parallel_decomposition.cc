#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/containers.h"
#include "utils/containers/all_of.h"
#include "utils/containers/extend.h"
#include "utils/containers/get_only.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/graph/serial_parallel/intermediate_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/serial_parallel_metrics.h"
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

bool is_empty(Node const &node) {
  return false;
}

bool is_empty(SerialSplit const &serial) {
  return all_of(serial.children, [](auto const &child) {
    return is_empty(widen<SerialParallelDecomposition>(child));
  });
}

bool is_empty(ParallelSplit const &parallel) {
  return all_of(parallel.children, [](auto const &child) {
    return is_empty(widen<SerialParallelDecomposition>(child));
  });
}

bool is_empty(SerialParallelDecomposition const &sp) {
  return sp.visit<bool>([](auto const &t) { return is_empty(t); });
}

size_t num_nodes(SerialParallelDecomposition const &sp) {
  return sum(values(get_node_frequency_map(sp)));
}

SerialParallelDecomposition serial_composition(
    std::vector<SerialParallelDecomposition> const &sp_compositions) {
  SerialSplit composition{};
  for (SerialParallelDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<SerialSplit>()) {
      extend(composition.children, sp_comp.get<SerialSplit>().children);
    } else if (sp_comp.has<ParallelSplit>()) {
      composition.children.push_back(sp_comp.get<ParallelSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.children.push_back(sp_comp.get<Node>());
    }
  }
  return SerialParallelDecomposition(composition);
}

SerialParallelDecomposition parallel_composition(
    std::unordered_set<SerialParallelDecomposition> const &sp_compositions) {
  ParallelSplit composition{};
  for (SerialParallelDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<ParallelSplit>()) {
      composition.children = set_union(composition.children,
                                       sp_comp.get<ParallelSplit>().children);
    } else if (sp_comp.has<SerialSplit>()) {
      composition.children.insert(sp_comp.get<SerialSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.children.insert(sp_comp.get<Node>());
    }
  }
  return SerialParallelDecomposition(composition);
}

} // namespace FlexFlow

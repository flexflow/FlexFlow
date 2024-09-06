#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/containers/generate_map.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/node/algorithms.h"
#include <unordered_map>

namespace FlexFlow {

enum class ExplorationStatus { NOT_EXPLORED, BEING_EXPLORED, FULLY_EXPLORED };

bool is_acyclic(DiGraphView const &g) {
  if (num_nodes(g) == 0) {
    return true; // vacuously true
  }

  std::unordered_map<Node, ExplorationStatus> status =
      generate_map(get_nodes(g), [](Node const &n) {
        return ExplorationStatus::NOT_EXPLORED;
      });

  // recursively explore a given node and all its successors: if, while
  // exploring, we find a node that was already being explored, then there is a
  // cycle
  std::function<bool(Node)> cycle_downstream_from_node =
      [&](Node const &n) -> bool {
    status[n] = ExplorationStatus::BEING_EXPLORED;

    for (Node const &successor : get_successors(g, n)) {
      if (status.at(successor) == ExplorationStatus::NOT_EXPLORED) {
        if (cycle_downstream_from_node(
                successor)) { // one of the descendants is part of a cycle
          return true;
        }
      } else if (status.at(successor) == ExplorationStatus::BEING_EXPLORED) {
        return true; // we're exploring a node we were already exploring: we
                     // have hit a cycle
      }
    }

    status[n] = ExplorationStatus::FULLY_EXPLORED;
    return false;
  };

  for (Node const &node : get_nodes(g)) {
    if (status.at(node) == ExplorationStatus::NOT_EXPLORED) {
      if (cycle_downstream_from_node(node)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace FlexFlow

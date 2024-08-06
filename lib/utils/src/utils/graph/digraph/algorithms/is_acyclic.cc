#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/containers/generate_map.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/traversal.h"
#include <unordered_map>

namespace FlexFlow {

enum class ExplorationStatus { NOT_EXPLORED, BEING_EXPLORED, FULLY_EXPLORED };

static bool
    dfs_is_acyclic(DiGraphView const &g,
                   Node const &n,
                   std::unordered_map<Node, ExplorationStatus> &status) {
  status[n] = ExplorationStatus::BEING_EXPLORED;

  for (Node const &successor : get_successors(g, n)) {
    if (status[successor] == ExplorationStatus::NOT_EXPLORED) {
      if (!dfs_is_acyclic(g, successor, status)) {
        return false;
      }
    } else if (status.at(successor) == ExplorationStatus::BEING_EXPLORED) {
      return false;
    }
  }

  status[n] = ExplorationStatus::FULLY_EXPLORED;
  return true;
}

bool is_acyclic(DiGraphView const &g) {
  if (num_nodes(g) == 0) {
    return true; // vacuously true
  }

  std::unordered_map<Node, ExplorationStatus> status =
      generate_map(get_nodes(g), [](Node const &n) {
        return ExplorationStatus::NOT_EXPLORED;
      });

  for (Node const &node : get_nodes(g)) {
    if (status.at(node) == ExplorationStatus::NOT_EXPLORED) {
      if (!dfs_is_acyclic(g, node, status)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace FlexFlow

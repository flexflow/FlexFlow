#include "utils/graph/undirected/algorithms/get_connected_components.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/hash/unordered_set.h"

namespace FlexFlow {

std::unordered_set<std::unordered_set<Node>>
    get_connected_components(UndirectedGraphView const &g) {
  std::unordered_set<std::unordered_set<Node>> components;
  std::unordered_set<Node> visited;

  for (Node const &node : get_nodes(g)) {
    std::unordered_set<Node> component =
        unordered_set_of(get_bfs_ordering(as_digraph(g), {node}));
    components.insert(component);
    visited = set_union(visited, component);
  }
  return components;
}

} // namespace FlexFlow

#ifndef _FLEXFLOW_UTILS_GRAPH_CONSTRUCTION_H
#define _FLEXFLOW_UTILS_GRAPH_CONSTRUCTION_H

#include "multidigraph.h"
#include "node.h"
#include <functional>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename G>
G make_multidigraph(std::size_t num_nodes,
                    std::function<std::unordered_set<MultiDiEdge>(
                        std::vector<Node> const &)> const &edges) {
  G g;
  std::vector<Node> nodes;
  for (std::size_t i = 0; i < num_nodes; i++) {
    nodes.push_back(g.add_node());
  }

  for (MultiDiEdge const &e : edges(nodes)) {
    g.add_edge(e);
  }
  return g;
}

} // namespace FlexFlow

#endif

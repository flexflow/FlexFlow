#include "utils/graph/multidigraph/algorithms/add_edges.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::vector<MultiDiEdge> add_edges(MultiDiGraph &g, std::vector<std::pair<Node, Node>> const &es) {
  return transform(es, 
                   [&](std::pair<Node, Node> const &p) { return g.add_edge(p.first, p.second); });
}

} // namespace FlexFlow

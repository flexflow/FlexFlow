#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/graph/node/algorithms.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(DiGraphView const &g) {
  return get_predecessors(g, get_nodes(g));
}

std::unordered_set<Node> get_predecessors(DiGraphView const &g, Node const &n) {
  return get_predecessors(g, std::unordered_set<Node>{n}).at(n);
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(DiGraphView const &g, std::unordered_set<Node> const &ns) {
  return map_values(get_incoming_edges(g, ns),
                    [](std::unordered_set<DirectedEdge> const &es) {
                      return transform(
                          es, [](DirectedEdge const &e) { return e.src; });
                    });
}


} // namespace FlexFlow

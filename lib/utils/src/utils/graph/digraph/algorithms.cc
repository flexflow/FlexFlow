#include "utils/graph/digraph/algorithms.h"
#include "utils/containers/group_by.h"
#include "utils/containers/set_minus.h"
#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/traversal.h"
#include "utils/graph/views/views.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_edges(DiGraphView const &g) {
  return g.query_edges(directed_edge_query_all());
}

std::unordered_set<DirectedEdge> get_incoming_edges(DiGraphView const &g,
                                                    Node const &n) {
  return g.query_edges(DirectedEdgeQuery{
      query_set<Node>::matchall(),
      query_set<Node>{n},
  });
}

std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_incoming_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  std::unordered_map<Node, std::unordered_set<DirectedEdge>> result =
      group_by(g.query_edges(DirectedEdgeQuery{
                   query_set<Node>::matchall(),
                   query_set<Node>{ns},
               }),
               [](DirectedEdge const &e) { return e.dst; });

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraphView const &g,
                                                    Node const &n) {
  return g.query_edges(DirectedEdgeQuery{
      query_set<Node>{n},
      query_set<Node>::matchall(),
  });
}

std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_outgoing_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  std::unordered_map<Node, std::unordered_set<DirectedEdge>> result =
      group_by(g.query_edges(DirectedEdgeQuery{
                   query_set<Node>::matchall(),
                   query_set<Node>{ns},
               }),
               [](DirectedEdge const &e) { return e.src; });

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

std::unordered_set<Node> get_sources(DiGraphView const &g) {
  std::unordered_set<Node> all_nodes = get_nodes(g);
  std::unordered_set<Node> with_incoming_edge =
      transform(get_edges(g), [](DirectedEdge const &e) { return e.dst; });

  return set_minus(all_nodes, with_incoming_edge);
}

std::unordered_set<Node> get_sinks(DiGraphView const &g) {
  return get_sources(flipped(g));
}

DiGraphView flipped(DiGraphView const &g) {
  return DiGraphView::create<FlippedView>(g);
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

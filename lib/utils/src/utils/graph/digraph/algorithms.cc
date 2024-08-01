#include "utils/graph/digraph/algorithms.h"
#include "utils/containers/group_by.h"
#include "utils/containers/map_values.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms/flipped.h"
#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/traversal.h"
#include "utils/graph/views/views.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_edges(DiGraphView const &g) {
  return g.query_edges(directed_edge_query_all());
}

std::unordered_set<Node> get_sinks(DiGraphView const &g) {
  return get_sources(flipped(g));
}

std::unordered_set<Node> get_sources(DiGraphView const &g) {
  std::unordered_set<Node> all_nodes = get_nodes(g);
  std::unordered_set<Node> with_incoming_edge =
      transform(get_edges(g), [](DirectedEdge const &e) { return e.dst; });

  return set_minus(all_nodes, with_incoming_edge);
}

} // namespace FlexFlow

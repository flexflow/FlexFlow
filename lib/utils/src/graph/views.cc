#include "utils/graph/views.h"
#include "utils/containers.h"

namespace FlexFlow {
namespace utils {

FlippedView::FlippedView(IDiGraphView const &g)
  : g(&g)
{ }

std::unordered_set<DirectedEdge> FlippedView::query_edges(DirectedEdgeQuery const &query) const {
  std::unordered_set<DirectedEdge> result = this->g->query_edges({query.dsts, query.srcs});
  return map_over_unordered_set<DirectedEdge, DirectedEdge>(flipped, result);
}

std::unordered_set<Node> FlippedView::query_nodes(NodeQuery const &query) const {
  return this->g->query_nodes(query);
}

DirectedEdge flipped(DirectedEdge const &e) {
  return {e.src, e.dst};  
}


FlippedView unsafe_view_as_flipped(IDiGraphView const &g) {
  return FlippedView(g);
}

}
}

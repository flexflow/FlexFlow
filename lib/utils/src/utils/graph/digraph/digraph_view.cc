#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::unordered_set<Node> DiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}

std::unordered_set<DirectedEdge>
    DiGraphView::query_edges(EdgeQuery const &query) const {
  return get_ptr().query_edges(query);
}

IDiGraphView const &DiGraphView::get_ptr() const {
  return *std::dynamic_pointer_cast<IDiGraphView const>(GraphView::ptr.get());
}

} // namespace FlexFlow

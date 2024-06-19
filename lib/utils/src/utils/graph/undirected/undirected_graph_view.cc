#include "utils/graph/undirected/undirected_graph_view.h"

namespace FlexFlow {

std::unordered_set<UndirectedEdge>
    UndirectedGraphView::query_edges(UndirectedEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

std::unordered_set<Node>
    UndirectedGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}

IUndirectedGraphView const &UndirectedGraphView::get_ptr() const {
  return *std::dynamic_pointer_cast<IUndirectedGraphView const>(
      GraphView::ptr.get());
}

} // namespace FlexFlow

#include "utils/graph/multidigraph/multidigraph_view.h"

namespace FlexFlow {

std::unordered_set<Node>
    MultiDiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_interface().query_nodes(q);
}

std::unordered_set<MultiDiEdge>
    MultiDiGraphView::query_edges(MultiDiEdgeQuery const &q) const {
  return this->get_interface().query_edges(q);
}

Node MultiDiGraphView::get_multidiedge_src(MultiDiEdge const &e) const {
  return this->get_interface().get_multidiedge_src(e);
}

Node MultiDiGraphView::get_multidiedge_dst(MultiDiEdge const &e) const {
  return this->get_interface().get_multidiedge_dst(e);
}

IMultiDiGraphView const &MultiDiGraphView::get_interface() const {
  return *std::dynamic_pointer_cast<IMultiDiGraphView const>(
      GraphView::ptr.get());
}

} // namespace FlexFlow

#include "utils/graph/node/graph_view.h"

namespace FlexFlow {

GraphView::GraphView(cow_ptr_t<IGraphView> ptr) : ptr(ptr) {}

std::unordered_set<Node> GraphView::query_nodes(NodeQuery const &g) const {
  return this->ptr->query_nodes(g);
}

bool is_ptr_equal(GraphView const &lhs, GraphView const &rhs) {
  return lhs.ptr == rhs.ptr;
}

} // namespace FlexFlow

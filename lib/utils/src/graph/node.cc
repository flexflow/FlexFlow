#include "utils/graph/node.h"
#include "utils/graph/cow_ptr_t.h"
#include <sstream>

namespace FlexFlow {

NodeQuery NodeQuery::all() {
  return {matchall<Node>()};
}

NodeQuery query_intersection(NodeQuery const &lhs, NodeQuery const &rhs) {

  std::unordered_set<Node> nodes;

  if (is_matchall(lhs.nodes) && !is_matchall(rhs.nodes)) {
    nodes = allowed_values(rhs.nodes);
  } else if (!is_matchall(lhs.nodes) && is_matchall(rhs.nodes)) {
    nodes = allowed_values(lhs.nodes);
  } else if (!is_matchall(lhs.nodes) && !is_matchall(rhs.nodes)) {
    nodes = allowed_values(query_intersection(lhs.nodes, rhs.nodes));
  }

  NodeQuery intersection_result = NodeQuery::all();
  intersection_result.nodes = nodes;

  return intersection_result;
}

std::unordered_set<Node> GraphView::query_nodes(NodeQuery const &g) const {
  return this->ptr->query_nodes(g);
}

// Set the shared_ptr's destructor to a nop so that effectively there is no
// ownership
GraphView
    GraphView::unsafe_create_without_ownership(IGraphView const &graphView) {
  std::shared_ptr<IGraphView const> ptr((&graphView),
                                        [](IGraphView const *) {});
  return GraphView(ptr);
}

Graph::Graph(cow_ptr_t<IGraph> _ptr)
  : ptr(std::move(_ptr))
{ 
  assert(this->ptr.get() != nullptr);
}

Node Graph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

} // namespace FlexFlow

#include "utils/graph/node.h"
#include <sstream>
#include "utils/containers.h"


namespace FlexFlow {

std::unordered_set<Node> GraphView::query_nodes(NodeQuery const &query) const {
  return ptr->query_nodes(query);
}

NodeQuery::NodeQuery(std::unordered_set<Node> const &nodes)
    : NodeQuery(tl::optional<std::unordered_set<Node>>{nodes}) {}

NodeQuery::NodeQuery(tl::optional<std::unordered_set<Node>> const &nodes)
    : nodes(nodes) {}

NodeQuery query_intersection(NodeQuery const & lhs, NodeQuery const & rhs){
    return  intersection(*lhs.nodes, *rhs.nodes) ;
    }

}// namespace FlexFlow

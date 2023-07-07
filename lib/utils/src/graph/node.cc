#include "utils/graph/node.h"
#include <sstream>

namespace FlexFlow {

NodeQuery::NodeQuery(std::unordered_set<Node> const &nodes)
    : NodeQuery(tl::optional<std::unordered_set<Node>>{nodes}) {}

NodeQuery::NodeQuery(tl::optional<std::unordered_set<Node>> const &nodes)
    : nodes(nodes) {}


} // namespace FlexFlow

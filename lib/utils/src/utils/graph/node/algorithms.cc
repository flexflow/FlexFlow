#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_query.h"

namespace FlexFlow {

std::unordered_set<Node> get_nodes(GraphView const &g) {
  return g.query_nodes(node_query_all());
}


} // namespace FlexFlow

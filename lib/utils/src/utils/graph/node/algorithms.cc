#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_query.h"

namespace FlexFlow {

std::unordered_set<Node> get_nodes(GraphView const &g) {
  return g.query_nodes(node_query_all());
}

size_t num_nodes(GraphView const &g) {
  return get_nodes(g).size();
}

bool empty(GraphView const &g) {
  return num_nodes(g) == 0;
}

} // namespace FlexFlow

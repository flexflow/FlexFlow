#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include "utils/containers/repeat.h"

namespace FlexFlow {

std::vector<Node> add_nodes(MultiDiGraph &g, int num_nodes) {
  return repeat(num_nodes, [&]() { return g.add_node(); });
}


} // namespace FlexFlow

#include "utils/graph/digraph/algorithms/get_node_with_greatest_topo_rank.h"
#include "utils/graph/digraph/algorithms/calculate_topo_rank.h"

namespace FlexFlow {

Node get_node_with_greatest_topo_rank(std::unordered_set<Node> const &nodes,
                                      DiGraphView const &g) {
  std::unordered_map<Node, int> topo_rank = calculate_topo_rank(g);
  return *std::max_element(nodes.cbegin(),
                           nodes.cend(),
                           [&topo_rank](Node const &lhs, Node const &rhs) {
                             return topo_rank.at(lhs) < topo_rank.at(rhs);
                           });
}

} // namespace FlexFlow

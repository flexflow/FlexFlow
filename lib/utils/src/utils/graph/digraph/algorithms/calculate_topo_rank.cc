#include "utils/graph/digraph/algorithms/calculate_topo_rank.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"

namespace FlexFlow {

std::unordered_map<Node, int> calculate_topo_rank(DiGraphView const &g) {
  std::vector<Node> topo_ordering = get_topological_ordering(g);
  std::unordered_map<Node, int> topo_rank;
  for (int i = 0; i < topo_ordering.size(); i++) {
    topo_rank[topo_ordering[i]] = i;
  }
  return topo_rank;
}

} // namespace FlexFlow

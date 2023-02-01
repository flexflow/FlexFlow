#include "utils/graph/algorithms.h"

using namespace FlexFlow::utils::graph;

std::unordered_set<Node> get_nodes(IMultiDiGraph const &g) {
  return g.query_nodes({});
}

std::vector<Node> topo_sort(IMultiDiGraph const &g) {
  return 
}

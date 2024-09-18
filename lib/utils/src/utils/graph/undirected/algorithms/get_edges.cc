#include "utils/graph/undirected/algorithms/get_edges.h"
#include "utils/graph/undirected/undirected_edge_query.h"

namespace FlexFlow {

std::unordered_set<UndirectedEdge> get_edges(UndirectedGraphView const &g) {
  return g.query_edges(undirected_edge_query_all());
}

} // namespace FlexFlow

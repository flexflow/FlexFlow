#include "utils/graph/undirected/algorithms/get_neighboring_nodes.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

std::unordered_set<Node> get_neighboring_nodes(UndirectedGraphView const &g, Node const &n) {
  std::unordered_set<UndirectedEdge> edges = g.query_edges(UndirectedEdgeQuery{query_set<Node>{n}});

  std::unordered_set<Node> result = set_union(transform(vector_of(edges), [](UndirectedEdge const &e) { return std::unordered_set{e.bigger, e.smaller}; }));
  result.erase(n);
  return result;
}

} // namespace FlexFlow

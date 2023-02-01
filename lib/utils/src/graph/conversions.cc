#include "utils/graph/conversions.h"
#include <algorithm>
#include <iterator>

namespace FlexFlow {
namespace utils {
namespace graph {

undirected::Edge to_undirected_edge(digraph::Edge const &e) {
  return {e.src, e.dst};
}

UndirectedGraphViewFromDirected::UndirectedGraphViewFromDirected(std::shared_ptr<IDiGraphView> const &directed) 
  : directed(directed)
{ }

std::unordered_set<undirected::Edge> UndirectedGraphViewFromDirected::query_edges(undirected::EdgeQuery const &undirected_query) const {
  digraph::EdgeQuery directed_query { undirected_query.nodes, undirected_query.nodes };
  std::unordered_set<digraph::Edge> const directed_edges = this->directed->query_edges(directed_query);

  return [&] {
    std::unordered_set<undirected::Edge> result;
    std::transform(directed_edges.cbegin(), directed_edges.cend(), 
                   std::inserter(result, result.begin()),
                   [](digraph::Edge const &e) { return to_undirected_edge(e); });
    return result;
  }();
}

std::unordered_set<Node> UndirectedGraphViewFromDirected::query_nodes(NodeQuery const &query) const {
  return this->directed->query_nodes(query);
}

UndirectedGraphViewFromDirected view_undirected(std::shared_ptr<IDiGraph> const &directed) {
  return UndirectedGraphViewFromDirected{directed};
}

}
}
}

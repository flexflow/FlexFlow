#include "utils/graph/conversions.h"
#include <algorithm>
#include <iterator>

using namespace FlexFlow::utils::graph;

UndirectedGraphViewFromDirected::UndirectedGraphViewFromDirected(std::shared_ptr<IDiGraphView> const &directed) 
  : directed(directed)
{ }

std::unordered_set<undirected::Edge> UndirectedGraphViewFromDirected::query_edges(undirected::EdgeQuery const &undirected_query) const {
  digraph::EdgeQuery directed_query { undirected_query.nodes, undirected_query.nodes };
  std::unordered_set<digraph::Edge> const directed_edges = this->directed->query_edges(directed_query);

  return [&] {
    std::unordered_set<undirected::Edge> result;
    std::transform(directed_edges.cbegin(), directed_edges.cend(), 
                   std::back_inserter(result),
                   to_undirected_edge);
    return result;
  }();
}

std::unordered_set<Node> UndirectedGraphViewFromDirected::query_nodes(NodeQuery const &query) const {
  return this->directed->query_nodes(query);
}

UndirectedGraphViewFromDirected view_undirected(std::shared_ptr<IDiGraph> const &directed) {
  return UndirectedGraphViewFromDirected{directed};
}


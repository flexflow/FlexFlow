#include "utils/graph/conversions.h"
#include <algorithm>
#include <iterator>

namespace FlexFlow {
namespace utils {
namespace graph {

undirected::Edge to_undirected_edge(digraph::Edge const &e) {
  return {e.src, e.dst};
}

ViewDiGraphAsUndirectedGraph::ViewDiGraphAsUndirectedGraph(std::shared_ptr<IDiGraphView> const &directed) 
  : directed(directed)
{ }

std::unordered_set<undirected::Edge> ViewDiGraphAsUndirectedGraph::query_edges(undirected::EdgeQuery const &undirected_query) const {
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

std::unordered_set<Node> ViewDiGraphAsUndirectedGraph::query_nodes(NodeQuery const &query) const {
  return this->directed->query_nodes(query);
}

ViewDiGraphAsUndirectedGraph unsafe_view_as_undirected(IDiGraphView const &directed) {
  return ViewDiGraphAsUndirectedGraph{directed};
}

ViewDiGraphAsUndirectedGraph view_as_undirected(std::shared_ptr<IDiGraph> const &directed) {
  return ViewDiGraphAsUndirectedGraph{directed};
}

ViewDiGraphAsMultiDiGraph unsafe_view_as_multidigraph(IDiGraphView const &directed) {
  return ViewDiGraphAsMultiDiGraph{directed};
}

ViewDiGraphAsMultiDiGraph view_as_multidigraph(std::shared_ptr<IDiGraphView> const &directed) {
  return ViewDiGraphAsMultiDiGraph{directed};
}

ViewMultiDiGraphAsDiGraph unsafe_view_as_digraph(IMultiDiGraphView const &multidi) {
  return ViewMultiDiGraphAsDiGraph{multidi}; 
}

ViewMultiDiGRaphAsDiGraph view_as_multidigraph(std::shared_ptr<IDiGraphView> const &undirected) {
  return View
}

}
}
}

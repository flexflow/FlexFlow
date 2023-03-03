#include "utils/graph/conversions.h"
#include <algorithm>
#include <iterator>

namespace FlexFlow {

UndirectedEdge to_undirected_edge(DirectedEdge const &e) {
  return {e.src, e.dst};
}

std::unordered_set<UndirectedEdge> to_undirected_edges(std::unordered_set<DirectedEdge> const &directed_edges) {
  std::unordered_set<UndirectedEdge> result;
  std::transform(directed_edges.cbegin(), directed_edges.cend(), 
                 std::inserter(result, result.begin()),
                 [](DirectedEdge const &e) { return to_undirected_edge(e); });
  return result;
}

UndirectedEdge to_undirected_edge(MultiDiEdge const &e) {
  return to_undirected_edge(to_directed_edge(e));
}

std::unordered_set<UndirectedEdge> to_undirected_edges(std::unordered_set<MultiDiEdge> const &multidi_edges) {
  return to_undirected_edges(to_directed_edges(multidi_edges));
}

std::unordered_set<DirectedEdge> to_directed_edges(UndirectedEdge const &e) {
  return {{e.smaller, e.bigger}, {e.bigger, e.smaller}};
}

std::unordered_set<DirectedEdge> to_directed_edges(std::unordered_set<UndirectedEdge> const &undirected_edges) {
  std::unordered_set<DirectedEdge> result;
  for (UndirectedEdge const &e : undirected_edges) {
    std::unordered_set<DirectedEdge> const for_e = to_directed_edges(e);
    result.insert(for_e.cbegin(), for_e.cend());
  }
  return result;
}

DirectedEdge to_directed_edge(MultiDiEdge const &e) {
  return {e.src, e.dst};
}

std::unordered_set<DirectedEdge> to_directed_edges(std::unordered_set<MultiDiEdge> const &multidi_edges) {
  std::unordered_set<DirectedEdge> result;
  std::transform(multidi_edges.cbegin(), multidi_edges.cend(), 
                 std::inserter(result, result.begin()),
                 [](MultiDiEdge const &e) { return to_directed_edge(e); });
  return result;
}

std::unordered_set<MultiDiEdge> to_multidigraph_edges(UndirectedEdge const &e) {
  return to_multidigraph_edges(to_directed_edges(e));
}

std::unordered_set<MultiDiEdge> to_multidigraph_edges(std::unordered_set<UndirectedEdge> const &undirected_edges) {
  std::unordered_set<MultiDiEdge> result;
  for (UndirectedEdge const &e : undirected_edges) {
    std::unordered_set<MultiDiEdge> const for_e = to_multidigraph_edges(e);
    result.insert(for_e.cbegin(), for_e.cend());
  }
  return result;
}

MultiDiEdge to_multidigraph_edge(DirectedEdge const &e) {
  return {e.src, e.dst, 0, 0};
}

std::unordered_set<MultiDiEdge> to_multidigraph_edges(std::unordered_set<DirectedEdge> const &digraph_edges) {
  std::unordered_set<MultiDiEdge> result;
  std::transform(digraph_edges.cbegin(), digraph_edges.cend(), 
                 std::inserter(result, result.begin()),
                 [](DirectedEdge const &e) { return to_multidigraph_edge(e); });
  return result;
}

ViewDiGraphAsUndirectedGraph::ViewDiGraphAsUndirectedGraph(IDiGraphView const &directed)
  : directed(&directed), shared(nullptr)
{ }

ViewDiGraphAsUndirectedGraph::ViewDiGraphAsUndirectedGraph(std::shared_ptr<IDiGraphView> const &directed) 
  : directed(directed.get()), shared(directed)
{ }

std::unordered_set<UndirectedEdge> ViewDiGraphAsUndirectedGraph::query_edges(UndirectedEdgeQuery const &undirected_query) const {
  DirectedEdgeQuery directed_query { undirected_query.nodes, undirected_query.nodes };
  std::unordered_set<DirectedEdge> const directed_edges = this->directed->query_edges(directed_query);
  return to_undirected_edges(directed_edges);
}

std::unordered_set<Node> ViewDiGraphAsUndirectedGraph::query_nodes(NodeQuery const &node_query) const {
  return this->directed->query_nodes(node_query);
}

ViewDiGraphAsMultiDiGraph::ViewDiGraphAsMultiDiGraph(IDiGraphView const &directed) 
  : directed(&directed), shared(nullptr)
{ }

ViewDiGraphAsMultiDiGraph::ViewDiGraphAsMultiDiGraph(std::shared_ptr<IDiGraphView> const &directed) 
  : directed(directed.get()), shared(directed)
{ }

std::unordered_set<MultiDiEdge> ViewDiGraphAsMultiDiGraph::query_edges(MultiDiEdgeQuery const &multidi_query) const {
  DirectedEdgeQuery directed_query { multidi_query.srcs, multidi_query.dsts };
  std::unordered_set<DirectedEdge> const directed_edges = this->directed->query_edges(directed_query);

  return [&] {
    std::unordered_set<MultiDiEdge> result;
    std::transform(directed_edges.begin(), directed_edges.cend(),
                   std::inserter(result, result.begin()),
                   [](DirectedEdge const &e) { return to_multidigraph_edge(e); });
    return result;
  }();
}

std::unordered_set<Node> ViewDiGraphAsMultiDiGraph::query_nodes(NodeQuery const &node_query) const {
  return this->directed->query_nodes(node_query);
}

ViewMultiDiGraphAsDiGraph::ViewMultiDiGraphAsDiGraph(IMultiDiGraphView const &multidi) 
  : multidi(&multidi), shared(nullptr)
{ }

ViewMultiDiGraphAsDiGraph::ViewMultiDiGraphAsDiGraph(std::shared_ptr<IMultiDiGraphView> const &multidi)
  : multidi(multidi.get()), shared(multidi)
{ }

std::unordered_set<DirectedEdge> ViewMultiDiGraphAsDiGraph::query_edges(DirectedEdgeQuery const &digraph_query) const {
  MultiDiEdgeQuery multidi_query { digraph_query.srcs, digraph_query.dsts };
  std::unordered_set<MultiDiEdge> const multidi_edges = this->multidi->query_edges(multidi_query);

  return [&] {
    std::unordered_set<DirectedEdge> result;
    std::transform(multidi_edges.cbegin(), multidi_edges.cend(), 
                   std::inserter(result, result.begin()),
                   [](MultiDiEdge const &e) { return to_directed_edge(e); });
    return result;
  }();
}

std::unordered_set<Node> ViewMultiDiGraphAsDiGraph::query_nodes(NodeQuery const &query) const {
  return this->multidi->query_nodes(query);
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

ViewMultiDiGraphAsDiGraph view_as_digraph(std::shared_ptr<IMultiDiGraph> const &multidi) {
  return ViewMultiDiGraphAsDiGraph{multidi};
}

}

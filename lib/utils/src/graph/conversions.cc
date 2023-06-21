#include "utils/graph/conversions.h"
#include "utils/containers.h"
#include <algorithm>
#include <iterator>

namespace FlexFlow {

UndirectedEdge to_undirected_edge(DirectedEdge const &e) {
  return {e.src, e.dst};
}

std::unordered_set<UndirectedEdge> to_undirected_edges(
    std::unordered_set<DirectedEdge> const &directed_edges) {
  std::unordered_set<UndirectedEdge> result;
  std::transform(directed_edges.cbegin(),
                 directed_edges.cend(),
                 std::inserter(result, result.begin()),
                 [](DirectedEdge const &e) { return to_undirected_edge(e); });
  return result;
}

UndirectedEdge to_undirected_edge(MultiDiEdge const &e) {
  return to_undirected_edge(to_directed_edge(e));
}

std::unordered_set<UndirectedEdge>
    to_undirected_edges(std::unordered_set<MultiDiEdge> const &multidi_edges) {
  return to_undirected_edges(to_directed_edges(multidi_edges));
}

std::unordered_set<DirectedEdge> to_directed_edges(UndirectedEdge const &e) {
  return {{e.smaller, e.bigger}, {e.bigger, e.smaller}};
}

std::unordered_set<DirectedEdge> to_directed_edges(
    std::unordered_set<UndirectedEdge> const &undirected_edges) {
  return flatmap_v2<DirectedEdge>(undirected_edges, to_directed_edges);
}

DirectedEdge to_directed_edge(MultiDiEdge const &e) {
  return {e.src, e.dst};
}

std::unordered_set<DirectedEdge>
    to_directed_edges(std::unordered_set<MultiDiEdge> const &multidi_edges) {
  return transform(multidi_edges, to_directed_edge);
}

ViewDiGraphAsUndirectedGraph::ViewDiGraphAsUndirectedGraph(
    IDiGraphView const &_directed)
    : directed(&_directed) {}

ViewDiGraphAsUndirectedGraph::ViewDiGraphAsUndirectedGraph(
    std::shared_ptr<IDiGraphView const> const &directed)
    : directed(directed.get()), shared(directed) {}

std::unordered_set<UndirectedEdge> ViewDiGraphAsUndirectedGraph::query_edges(
    UndirectedEdgeQuery const &undirected_query) const {
  DirectedEdgeQuery directed_query{undirected_query.nodes,
                                   undirected_query.nodes};
  std::unordered_set<DirectedEdge> const directed_edges =
      this->directed->query_edges(directed_query);
  return to_undirected_edges(directed_edges);
}

std::unordered_set<Node> ViewDiGraphAsUndirectedGraph::query_nodes(
    NodeQuery const &node_query) const {
  return this->directed->query_nodes(node_query);
}

ViewDiGraphAsMultiDiGraph::ViewDiGraphAsMultiDiGraph(
    IDiGraphView const &directed)
    : directed(&directed), shared(nullptr) {}

ViewDiGraphAsMultiDiGraph::ViewDiGraphAsMultiDiGraph(
    std::shared_ptr<IDiGraphView> const &directed)
    : directed(directed.get()), shared(directed) {}

std::unordered_set<MultiDiEdge> ViewDiGraphAsMultiDiGraph::query_edges(
    MultiDiEdgeQuery const &multidi_query) const {
  DirectedEdgeQuery directed_query{multidi_query.srcs, multidi_query.dsts};

  std::unordered_set<DirectedEdge> const directed_edges =
      this->directed->query_edges(directed_query);

  return transform(directed_edges, [](DirectedEdge const &e) {
    return MultiDiEdge(e.src, e.dst, NodePort(0), NodePort(0));
  });
}

std::unordered_set<Node>
    ViewDiGraphAsMultiDiGraph::query_nodes(NodeQuery const &node_query) const {
  return this->directed->query_nodes(node_query);
}

ViewMultiDiGraphAsDiGraph::ViewMultiDiGraphAsDiGraph(
    IMultiDiGraphView const &multidi)
    : multidi(&multidi) {}

ViewMultiDiGraphAsDiGraph::ViewMultiDiGraphAsDiGraph(
    std::shared_ptr<IMultiDiGraphView const> const &multidi)
    : multidi(multidi.get()), shared(multidi) {}

std::unordered_set<DirectedEdge> ViewMultiDiGraphAsDiGraph::query_edges(
    DirectedEdgeQuery const &digraph_query) const {
  MultiDiEdgeQuery multidi_query{digraph_query.srcs, digraph_query.dsts};
  std::unordered_set<MultiDiEdge> const multidi_edges =
      this->multidi->query_edges(multidi_query);

  return [&] {
    std::unordered_set<DirectedEdge> result;
    std::transform(multidi_edges.cbegin(),
                   multidi_edges.cend(),
                   std::inserter(result, result.begin()),
                   [](MultiDiEdge const &e) { return to_directed_edge(e); });
    return result;
  }();
}

std::unordered_set<Node>
    ViewMultiDiGraphAsDiGraph::query_nodes(NodeQuery const &query) const {
  return this->multidi->query_nodes(query);
}

std::unique_ptr<IUndirectedGraphView>
    unsafe_view_as_undirected(IDiGraphView const &directed) {
  return std::unique_ptr<IUndirectedGraphView>(
      new ViewDiGraphAsUndirectedGraph{directed});
}

std::unique_ptr<IUndirectedGraphView>
    view_as_undirected(std::shared_ptr<IDiGraph> const &directed) {
  return std::unique_ptr<IUndirectedGraphView>(
      new ViewDiGraphAsUndirectedGraph{directed});
}

std::unique_ptr<IMultiDiGraphView>
    unsafe_view_as_multidigraph(IDiGraphView const &directed) {
  return std::unique_ptr<IMultiDiGraphView>(
      new ViewDiGraphAsMultiDiGraph(directed));
}

std::unique_ptr<IMultiDiGraphView>
    view_as_multidigraph(std::shared_ptr<IDiGraphView> const &directed) {
  return std::unique_ptr<IMultiDiGraphView>(
      new ViewDiGraphAsMultiDiGraph(directed));
}

std::unique_ptr<IDiGraphView>
    unsafe_view_as_digraph(IMultiDiGraphView const &multidi) {
  return std::unique_ptr<IDiGraphView>(new ViewMultiDiGraphAsDiGraph{multidi});
}

std::unique_ptr<IDiGraphView>
    view_as_digraph(std::shared_ptr<IMultiDiGraph> const &multidi) {
  return std::unique_ptr<IDiGraphView>(new ViewMultiDiGraphAsDiGraph{multidi});
}

UndirectedGraphView view_as_undirected(DiGraphView const & g) {
  GraphView graphView = static_cast<GraphView>(g);
  IGraphView const *graphViewPtr = graphView.unsafe();
  std::shared_ptr<IUndirectedGraphView const> undirectedPtr = std::dynamic_pointer_cast<IUndirectedGraphView const>(std::shared_ptr<IGraphView const>(const_cast<IGraphView *>(graphViewPtr)));
  return UndirectedGraphView(undirectedPtr);//IUndirectedGraphView :public IGraphView
}

MultiDiGraphView view_as_multidigraph(DiGraphView const & g) {
  GraphView graphView = static_cast<GraphView>(g);
  IGraphView const *graphViewPtr = graphView.unsafe();
  std::shared_ptr<IMultiDiGraphView const> multidigraphPtr = std::dynamic_pointer_cast<IMultiDiGraphView const>(std::shared_ptr<IGraphView const>(const_cast<IGraphView *>(graphViewPtr)));
  return MultiDiGraphView(multidigraphPtr);//IMultiDiGraphView : public IGraphView
}

DiGraphView view_as_digraph(MultiDiGraphView const & g) {
  GraphView graphView = static_cast<GraphView>(g);
  IGraphView const *graphViewPtr = graphView.unsafe();
  std::shared_ptr<IDiGraphView const> digraphPtr = std::dynamic_pointer_cast<IDiGraphView const>(std::shared_ptr<IGraphView const>(const_cast<IGraphView *>(graphViewPtr)));//IDiGraphView : public IGraphView
  return DiGraphView(digraphPtr);
}

} // namespace FlexFlow

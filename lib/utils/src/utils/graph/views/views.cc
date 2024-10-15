#include "utils/graph/views/views.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/transform.h"
#include "utils/disjoint_set.h"
#include "utils/exception.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/query_set.h"
#include "utils/graph/undirected/undirected_edge_query.h"
namespace FlexFlow {

UndirectedSubgraphView::UndirectedSubgraphView(
    UndirectedGraphView const &g,
    std::unordered_set<Node> const &subgraph_nodes)
    : g(g), subgraph_nodes(subgraph_nodes) {}

UndirectedSubgraphView *UndirectedSubgraphView::clone() const {
  return new UndirectedSubgraphView(g, subgraph_nodes);
}

std::unordered_set<UndirectedEdge> UndirectedSubgraphView::query_edges(
    UndirectedEdgeQuery const &query) const {
  UndirectedEdgeQuery subgraph_query =
      UndirectedEdgeQuery{this->subgraph_nodes};
  return this->g.query_edges(query_intersection(query, subgraph_query));
}

std::unordered_set<Node>
    UndirectedSubgraphView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(
      query_intersection(query, NodeQuery{this->subgraph_nodes}));
}

DiSubgraphView::DiSubgraphView(DiGraphView const &g,
                               std::unordered_set<Node> const &subgraph_nodes)
    : g(g), subgraph_nodes(subgraph_nodes) {}

std::unordered_set<DirectedEdge>
    DiSubgraphView::query_edges(DirectedEdgeQuery const &query) const {
  DirectedEdgeQuery subgraph_query =
      DirectedEdgeQuery{this->subgraph_nodes, this->subgraph_nodes};
  return this->g.query_edges(query_intersection(query, subgraph_query));
}

std::unordered_set<Node>
    DiSubgraphView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(
      query_intersection(query, NodeQuery{this->subgraph_nodes}));
}

DiSubgraphView *DiSubgraphView::clone() const {
  return new DiSubgraphView(g, subgraph_nodes);
}

UndirectedGraphView
    view_subgraph(UndirectedGraphView const &g,
                  std::unordered_set<Node> const &subgraph_nodes) {
  return UndirectedGraphView::create<UndirectedSubgraphView>(g, subgraph_nodes);
}

DiGraphView view_subgraph(DiGraphView const &g,
                          std::unordered_set<Node> const &subgraph_nodes) {
  return DiGraphView::create<DiSubgraphView>(g, subgraph_nodes);
}

UndirectedEdge to_undirected_edge(DirectedEdge const &e) {
  return UndirectedEdge{{e.src, e.dst}};
}

std::unordered_set<UndirectedEdge> to_undirected_edges(
    std::unordered_set<DirectedEdge> const &directed_edges) {
  return transform(directed_edges,
                   [](DirectedEdge const &e) { return to_undirected_edge(e); });
}

std::unordered_set<DirectedEdge> to_directed_edges(UndirectedEdge const &e) {
  return std::unordered_set<DirectedEdge>{
      DirectedEdge{e.endpoints.min(), e.endpoints.max()},
      DirectedEdge{e.endpoints.max(), e.endpoints.min()}};
}

std::unordered_set<DirectedEdge> to_directed_edges(
    std::unordered_set<UndirectedEdge> const &undirected_edges) {
  return flatmap_v2<DirectedEdge>(undirected_edges, to_directed_edges);
}

ViewDiGraphAsUndirectedGraph::ViewDiGraphAsUndirectedGraph(DiGraphView const &g)
    : g(g) {}

std::unordered_set<UndirectedEdge> ViewDiGraphAsUndirectedGraph::query_edges(
    UndirectedEdgeQuery const &undirected_query) const {
  DirectedEdgeQuery q1{undirected_query.nodes, query_set<Node>::matchall()};
  DirectedEdgeQuery q2{query_set<Node>::matchall(), undirected_query.nodes};
  return to_undirected_edges(
      set_union(this->g.query_edges(q1), this->g.query_edges(q2)));
}

std::unordered_set<Node> ViewDiGraphAsUndirectedGraph::query_nodes(
    NodeQuery const &node_query) const {
  return this->g.query_nodes(node_query);
}

ViewDiGraphAsUndirectedGraph *ViewDiGraphAsUndirectedGraph::clone() const {
  return new ViewDiGraphAsUndirectedGraph(g);
}

ViewUndirectedGraphAsDiGraph::ViewUndirectedGraphAsDiGraph(
    UndirectedGraphView const &g)
    : g(g) {}

ViewUndirectedGraphAsDiGraph *ViewUndirectedGraphAsDiGraph::clone() const {
  return new ViewUndirectedGraphAsDiGraph(g);
}

std::unordered_set<DirectedEdge> ViewUndirectedGraphAsDiGraph::query_edges(
    DirectedEdgeQuery const &q) const {
  std::unordered_set<UndirectedEdge> undirected_edges =
      set_union(g.query_edges(UndirectedEdgeQuery{q.srcs}),
                g.query_edges(UndirectedEdgeQuery{q.dsts}));
  std::unordered_set<DirectedEdge> directed_edges =
      flatmap(undirected_edges,
              [](UndirectedEdge const &e) { return to_directed_edges(e); });
  return filter(directed_edges,
                [&](DirectedEdge const &e) { return matches_edge(q, e); });
}

std::unordered_set<Node>
    ViewUndirectedGraphAsDiGraph::query_nodes(NodeQuery const &q) const {
  return g.query_nodes(q);
}

} // namespace FlexFlow

#include "utils/graph/views/views.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/transform.h"
#include "utils/disjoint_set.h"
#include "utils/exception.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_query.h"
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

JoinedNodeView::JoinedNodeView(GraphView const &lhs, GraphView const &rhs) {
  for (Node const &n : get_nodes(lhs)) {
    this->mapping.equate(JoinNodeKey{n, LRDirection::LEFT},
                         this->node_source.new_node());
  }
  for (Node const &n : get_nodes(rhs)) {
    this->mapping.equate(JoinNodeKey{n, LRDirection::RIGHT},
                         this->node_source.new_node());
  }
}

std::unordered_set<Node>
    JoinedNodeView::query_nodes(NodeQuery const &query) const {
  // TODO @lockshaw this is going to be reimplemented in 984, so don't bother
  // fixing it for now
  NOT_IMPLEMENTED();
}

std::pair<std::unordered_set<Node>, std::unordered_set<Node>>
    JoinedNodeView::trace_nodes(std::unordered_set<Node> const &nodes) const {
  std::unordered_set<Node> left_nodes, right_nodes;

  for (Node const &n : nodes) {
    JoinNodeKey k = this->at_node(n);
    if (k.direction == LRDirection::LEFT) {
      left_nodes.insert(k.node);
    } else {
      assert(k.direction == LRDirection::RIGHT);
      right_nodes.insert(k.node);
    }
  }

  return {left_nodes, right_nodes};
}

Node JoinedNodeView::at_join_key(JoinNodeKey const &k) const {
  return this->mapping.at_l(k);
}

JoinNodeKey JoinedNodeView::at_node(Node const &n) const {
  return this->mapping.at_r(n);
}

JoinedUndirectedGraphView::JoinedUndirectedGraphView(
    UndirectedGraphView const &lhs, UndirectedGraphView const &rhs)
    : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs) {}

std::unordered_set<Node>
    JoinedUndirectedGraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<UndirectedEdge> JoinedUndirectedGraphView::query_edges(
    UndirectedEdgeQuery const &query) const {
  std::unordered_set<Node> nodes = this->query_nodes(NodeQuery{query.nodes});
  std::unordered_set<Node> left_nodes, right_nodes;
  for (Node const &n : nodes) {
    JoinNodeKey k = this->joined_nodes.at_node(n);
    if (k.direction == LRDirection::LEFT) {
      left_nodes.insert(k.node);
    } else {
      assert(k.direction == LRDirection::RIGHT);
      right_nodes.insert(k.node);
    }
  }

  std::unordered_set<UndirectedEdge> result;
  for (UndirectedEdge const &e :
       this->lhs.query_edges(UndirectedEdgeQuery{left_nodes})) {
    result.insert(this->fix_lhs_edge(e));
  }
  for (UndirectedEdge const &e :
       this->rhs.query_edges(UndirectedEdgeQuery{right_nodes})) {
    result.insert(this->fix_rhs_edge(e));
  }

  return result;
}

UndirectedEdge
    JoinedUndirectedGraphView::fix_lhs_edge(UndirectedEdge const &e) const {
  return UndirectedEdge{{this->joined_nodes.at_join_key(
                             JoinNodeKey{e.endpoints.min(), LRDirection::LEFT}),
                         this->joined_nodes.at_join_key(JoinNodeKey{
                             e.endpoints.max(), LRDirection::LEFT})}};
}

UndirectedEdge
    JoinedUndirectedGraphView::fix_rhs_edge(UndirectedEdge const &e) const {
  return UndirectedEdge{{this->joined_nodes.at_join_key(JoinNodeKey{
                             e.endpoints.min(), LRDirection::RIGHT}),
                         this->joined_nodes.at_join_key(JoinNodeKey{
                             e.endpoints.max(), LRDirection::RIGHT})}};
}

JoinedDigraphView::JoinedDigraphView(DiGraphView const &lhs,
                                     DiGraphView const &rhs)
    : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs) {}

JoinedDigraphView *JoinedDigraphView::clone() const {
  return new JoinedDigraphView(lhs, rhs);
}

std::unordered_set<Node>
    JoinedDigraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<DirectedEdge>
    JoinedDigraphView::query_edges(DirectedEdgeQuery const &query) const {

  std::unordered_set<Node> srcs = this->query_nodes(NodeQuery{query.srcs});
  std::unordered_set<Node> dsts = this->query_nodes(NodeQuery{query.dsts});
  auto traced_srcs = this->joined_nodes.trace_nodes(srcs);
  auto traced_dsts = this->joined_nodes.trace_nodes(dsts);
  DirectedEdgeQuery left_query =
      DirectedEdgeQuery{traced_srcs.first, traced_dsts.first};
  DirectedEdgeQuery right_query =
      DirectedEdgeQuery{traced_srcs.second, traced_dsts.second};

  std::unordered_set<DirectedEdge> result;
  for (DirectedEdge const &e : this->lhs.query_edges(left_query)) {
    result.insert(this->fix_lhs_edge(e));
  }
  for (DirectedEdge const &e : this->rhs.query_edges(right_query)) {
    result.insert(this->fix_rhs_edge(e));
  }

  return result;
}

DirectedEdge JoinedDigraphView::fix_lhs_edge(DirectedEdge const &e) const {
  return DirectedEdge{
      this->joined_nodes.at_join_key(JoinNodeKey{e.src, LRDirection::LEFT}),
      this->joined_nodes.at_join_key(JoinNodeKey{e.dst, LRDirection::LEFT})};
}

DirectedEdge JoinedDigraphView::fix_rhs_edge(DirectedEdge const &e) const {
  return DirectedEdge{
      this->joined_nodes.at_join_key(JoinNodeKey{e.src, LRDirection::RIGHT}),
      this->joined_nodes.at_join_key(JoinNodeKey{e.dst, LRDirection::RIGHT})};
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

JoinedUndirectedGraphView *JoinedUndirectedGraphView::clone() const {
  return new JoinedUndirectedGraphView(lhs, rhs);
}

} // namespace FlexFlow

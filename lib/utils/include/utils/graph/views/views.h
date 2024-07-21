#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_VIEWS_VIEWS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_VIEWS_VIEWS_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/undirected/undirected_graph_view.h"
#include "utils/graph/views/join_node_key.dtg.h"

namespace FlexFlow {

struct FlippedView : public IDiGraphView {
public:
  FlippedView() = delete;
  explicit FlippedView(DiGraphView const &);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  FlippedView *clone() const override;

private:
  DiGraphView g;
};

struct UndirectedSubgraphView : public IUndirectedGraphView {
public:
  UndirectedSubgraphView() = delete;
  UndirectedSubgraphView(UndirectedGraphView const &,
                         std::unordered_set<Node> const &);

  std::unordered_set<UndirectedEdge>
      query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  UndirectedSubgraphView *clone() const override;

private:
  UndirectedGraphView g;
  std::unordered_set<Node> subgraph_nodes;
};

struct DiSubgraphView : public IDiGraphView {
public:
  DiSubgraphView() = delete;
  DiSubgraphView(DiGraphView const &, std::unordered_set<Node> const &);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  DiSubgraphView *clone() const override;

private:
  DiGraphView g;
  std::unordered_set<Node> subgraph_nodes;
};

struct JoinedNodeView {
public:
  JoinedNodeView() = delete;
  explicit JoinedNodeView(GraphView const &lhs, GraphView const &rhs);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::pair<std::unordered_set<Node>, std::unordered_set<Node>>
      trace_nodes(std::unordered_set<Node> const &) const;

  Node at_join_key(JoinNodeKey const &) const;
  JoinNodeKey at_node(Node const &) const;

private:
  bidict<JoinNodeKey, Node> mapping;
  NodeSource node_source;
};

struct JoinedUndirectedGraphView : public IUndirectedGraphView {
public:
  JoinedUndirectedGraphView() = delete;
  explicit JoinedUndirectedGraphView(UndirectedGraphView const &lhs,
                                     UndirectedGraphView const &rhs);

  std::unordered_set<UndirectedEdge>
      query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  JoinedUndirectedGraphView *clone() const override;

private:
  UndirectedEdge fix_lhs_edge(UndirectedEdge const &) const;
  UndirectedEdge fix_rhs_edge(UndirectedEdge const &) const;

private:
  UndirectedGraphView lhs;
  UndirectedGraphView rhs;
  JoinedNodeView joined_nodes;
};

struct JoinedDigraphView : virtual public IDiGraphView {
public:
  JoinedDigraphView() = delete;
  explicit JoinedDigraphView(DiGraphView const &lhs, DiGraphView const &rhs);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  JoinedNodeView const &joined_nodes_view() const;

  JoinedDigraphView *clone() const override;

private:
  DirectedEdge fix_lhs_edge(DirectedEdge const &) const;
  DirectedEdge fix_rhs_edge(DirectedEdge const &) const;

private:
  DiGraphView lhs;
  DiGraphView rhs;
  JoinedNodeView joined_nodes;
};

struct AddDirectedEdgesView : public IDiGraphView {
public:
  AddDirectedEdgesView() = delete;

  explicit AddDirectedEdgesView(DiGraphView const &g,
                                std::unordered_set<DirectedEdge> const &edges);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  AddDirectedEdgesView *clone() const override;

private:
  DiGraphView g;
  std::unordered_set<DirectedEdge> edges;
};

struct SingleSourceNodeView : public IDiGraphView {
public:
  SingleSourceNodeView() = delete;

  explicit SingleSourceNodeView(DiGraphView const &g) : g(g) {}

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  SingleSourceNodeView *clone() const override;

private:
  DiGraphView g;
  std::optional<AdjacencyDiGraph> singleton_src;
  std::optional<JoinedDigraphView> joined_view;
  std::unique_ptr<AddDirectedEdgesView> added_edges_view;
};

UndirectedEdge to_undirected_edge(DirectedEdge const &);
std::unordered_set<UndirectedEdge>
    to_undirected_edges(std::unordered_set<DirectedEdge> const &);

std::unordered_set<DirectedEdge> to_directed_edges(UndirectedEdge const &);
std::unordered_set<DirectedEdge>
    to_directed_edges(std::unordered_set<UndirectedEdge> const &);

struct ViewDiGraphAsUndirectedGraph : public IUndirectedGraphView {
public:
  explicit ViewDiGraphAsUndirectedGraph(DiGraphView const &);

  std::unordered_set<UndirectedEdge>
      query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  ViewDiGraphAsUndirectedGraph *clone() const override;

private:
  DiGraphView g;
};

struct ViewUndirectedGraphAsDiGraph : public IDiGraphView {
public:
  explicit ViewUndirectedGraphAsDiGraph(UndirectedGraphView const &);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  ViewUndirectedGraphAsDiGraph *clone() const override;

private:
  UndirectedGraphView g;
};

DirectedEdge flipped(DirectedEdge const &);

std::unordered_map<Node, Node>
    flatten_contraction(std::unordered_map<Node, Node> const &);

template <typename Impl, typename View>
Impl materialize_view(View const &g) {
  Impl result;
  for (Node const &n : get_nodes(g)) {
    result.add_node_unsafe(n);
  }
  for (auto const &e : get_edges(g)) {
    result.add_edge(e);
  }
  return result;
}

template <typename Impl>
Impl materialize_undirected_graph_view(IUndirectedGraphView const &g) {
  return materialize_view<Impl, IUndirectedGraphView>(g);
}

template <typename Impl>
Impl materialize_digraph_view(IDiGraphView const &g) {
  return materialize_view<Impl, IDiGraphView>(g);
}

} // namespace FlexFlow

#endif

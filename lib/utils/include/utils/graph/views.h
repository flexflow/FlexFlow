#ifndef _FLEXFLOW_UTILS_GRAPH_VIEWS_H
#define _FLEXFLOW_UTILS_GRAPH_VIEWS_H

#include "adjacency_digraph.h"
#include "digraph.h"
#include "multidiedge.h"
#include "multidigraph.h"
#include "node.h"
#include "open_graphs.h"
#include "tl/optional.hpp"
#include "undirected.h"
#include "utils/bidict.h"
#include "utils/visitable.h"
#include <memory>
#include <vector>

namespace FlexFlow {

struct FlippedView : public IDiGraphView {
public:
  FlippedView() = delete;
  explicit FlippedView(DiGraphView const &);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

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

private:
  DiGraphView g;
  std::unordered_set<Node> subgraph_nodes;
};

struct MultiDiSubgraphView : public IMultiDiGraphView {
public:
  MultiDiSubgraphView() = delete;
  explicit MultiDiSubgraphView(MultiDiGraphView const &,
                               std::unordered_set<Node> const &);

  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  MultiDiGraphView g;
  std::unordered_set<Node> subgraph_nodes;
};

struct NodeSource {
public:
  NodeSource() = default;

  Node fresh_node();

private:
  std::size_t next_node_idx = 0;
};

} // namespace FlexFlow

namespace FlexFlow {

struct JoinedUndirectedGraphView : public IUndirectedGraphView {
public:
  JoinedUndirectedGraphView() = delete;
  explicit JoinedUndirectedGraphView(UndirectedGraphView const &lhs,
                                     UndirectedGraphView const &rhs);

  std::unordered_set<UndirectedEdge>
      query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  UndirectedEdge fix_lhs_edge(UndirectedEdge const &) const;
  UndirectedEdge fix_rhs_edge(UndirectedEdge const &) const;

private:
  UndirectedGraphView lhs;
  UndirectedGraphView rhs;
  DuplicatedGraphView node_view;
};

struct JoinedDigraphView : public IDiGraphView {
public:
  JoinedDigraphView() = delete;
  explicit JoinedDigraphView(DiGraphView const &lhs, DiGraphView const &rhs);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  JoinedNodeView const &joined_nodes_view() const;

private:
  DirectedEdge fix_lhs_edge(DirectedEdge const &) const;
  DirectedEdge fix_rhs_edge(DirectedEdge const &) const;

private:
  DiGraphView lhs;
  DiGraphView rhs;
  DuplicatedGraphView node_view;
};

struct JoinedMultiDigraphView : public IMultiDiGraphView {
public:
  JoinedMultiDigraphView() = delete;
  JoinedMultiDigraphView(MultiDiGraphView const &lhs,
                         MultiDiGraphView const &rhs);

  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  JoinedNodeView const &joined_nodes_view() const;

private:
  MultiDiEdge fix_lhs_edge(MultiDiEdge const &) const;
  MultiDiEdge fix_rhs_edge(MultiDiEdge const &) const;

private:
  MultiDiGraphView lhs;
  MultiDiGraphView rhs;
  DuplicatedGraphView node_view;
};

struct AddDirectedEdgesView : public IDiGraphView {
public:
  AddDirectedEdgesView() = delete;

  explicit AddDirectedEdgesView(DiGraphView const &g,
                                std::unordered_set<DirectedEdge> const &edges);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

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

private:
  DiGraphView g;
  optional<AdjacencyDiGraph> singleton_src;
  optional<JoinedDigraphView> joined_view;
  std::unique_ptr<AddDirectedEdgesView> added_edges_view;
};

struct ContractNodeView : public IDiGraphView {
  ContractNodeView() = delete;
  explicit ContractNodeView(DiGraphView const &,
                            Node const &removed,
                            Node const &into);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  DirectedEdge fix_edge(DirectedEdge const &) const;

private:
  DiGraphView g;
  Node from, to;
};

struct OpenMultiDiSubgraphView : public IOpenMultiDiGraphView {
public:
  OpenMultiDiSubgraphView() = delete;
  explicit OpenMultiDiSubgraphView(OpenMultiDiGraphView const &,
                                   std::unordered_set<Node> const &);

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  OpenMultiDiGraphView g;
};

UndirectedEdge to_undirected_edge(DirectedEdge const &);
std::unordered_set<UndirectedEdge>
    to_undirected_edges(std::unordered_set<DirectedEdge> const &);
UndirectedEdge to_undirected_edge(MultiDiEdge const &);
std::unordered_set<UndirectedEdge>
    to_undirected_edges(std::unordered_set<MultiDiEdge> const &);

std::unordered_set<DirectedEdge> to_directed_edges(UndirectedEdge const &);
std::unordered_set<DirectedEdge>
    to_directed_edges(std::unordered_set<UndirectedEdge> const &);
DirectedEdge to_directed_edge(MultiDiEdge const &);
std::unordered_set<DirectedEdge>
    to_directed_edges(std::unordered_set<MultiDiEdge> const &);

struct ViewDiGraphAsUndirectedGraph : public IUndirectedGraphView {
public:
  explicit ViewDiGraphAsUndirectedGraph(DiGraphView const &);

  std::unordered_set<UndirectedEdge>
      query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  DiGraphView g;
};

struct ViewDiGraphAsMultiDiGraph : public IMultiDiGraphView {
public:
  explicit ViewDiGraphAsMultiDiGraph(DiGraphView const &);

  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  DiGraphView g;
};

struct ViewMultiDiGraphAsDiGraph : public IDiGraphView {
public:
  ViewMultiDiGraphAsDiGraph() = delete;
  explicit ViewMultiDiGraphAsDiGraph(MultiDiGraphView const &);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  MultiDiGraphView g;
};

struct ViewOpenMultiDiGraphAsMultiDiGraph : public IMultiDiGraphView {
public:
  ViewOpenMultiDiGraphAsMultiDiGraph() = delete;
  explicit ViewOpenMultiDiGraphAsMultiDiGraph(OpenMultiDiGraphView const &);

  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  OpenMultiDiGraphView const &g;
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

template <typename Impl>
Impl materialize_multidigraph_view(IMultiDiGraphView const &g) {
  return materialize_view<Impl, IMultiDiGraphView>(g);
}

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::JoinNodeKey, node, direction);

#endif

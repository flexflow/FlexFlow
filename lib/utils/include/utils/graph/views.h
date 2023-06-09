#ifndef _FLEXFLOW_UTILS_GRAPH_VIEWS_H
#define _FLEXFLOW_UTILS_GRAPH_VIEWS_H

#include "adjacency_digraph.h"
#include "digraph.h"
#include "multidigraph.h"
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
  explicit FlippedView(IDiGraphView const &);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  IDiGraphView const &g;
};

struct UndirectedSubgraphView : public IUndirectedGraphView {
public:
  UndirectedSubgraphView() = delete;
  UndirectedSubgraphView(maybe_owned_ref<IUndirectedGraphView const>,
                         std::unordered_set<Node> const &);

  std::unordered_set<UndirectedEdge>
      query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  maybe_owned_ref<IUndirectedGraphView const> g;
  std::unordered_set<Node> subgraph_nodes;
};

struct DiSubgraphView : public IDiGraphView {
public:
  DiSubgraphView() = delete;
  DiSubgraphView(maybe_owned_ref<IDiGraphView const>,
                 std::unordered_set<Node> const &);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  maybe_owned_ref<IDiGraphView const> g;
  std::unordered_set<Node> subgraph_nodes;
};

struct MultiDiSubgraphView : public IMultiDiGraphView {
public:
  MultiDiSubgraphView() = delete;
  explicit MultiDiSubgraphView(maybe_owned_ref<IMultiDiGraphView const>,
                               std::unordered_set<Node> const &);

  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  maybe_owned_ref<IMultiDiGraphView const> g;
  std::unordered_set<Node> subgraph_nodes;
};

struct NodeSource {
public:
  NodeSource() = default;

  Node fresh_node();

private:
  std::size_t next_node_idx = 0;
};

enum class LRDirection { LEFT, RIGHT };

struct JoinNodeKey {
  JoinNodeKey() = delete;
  JoinNodeKey(Node const &, LRDirection);

  bool operator==(JoinNodeKey const &) const;
  bool operator<(JoinNodeKey const &) const;

  Node node;
  LRDirection direction;
};

} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::JoinNodeKey> {
  std::size_t operator()(::FlexFlow::JoinNodeKey const &key) const {
    return std::hash<size_t>{}(static_cast<size_t>(key.node));
  }
};
} // namespace std

namespace FlexFlow {

struct JoinedNodeView {
public:
  JoinedNodeView() = delete;
  explicit JoinedNodeView(IGraphView const &lhs, IGraphView const &rhs);

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
  explicit JoinedUndirectedGraphView(
      maybe_owned_ref<IUndirectedGraphView const> lhs,
      maybe_owned_ref<IUndirectedGraphView const> rhs);

  std::unordered_set<UndirectedEdge>
      query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  UndirectedEdge fix_lhs_edge(UndirectedEdge const &) const;
  UndirectedEdge fix_rhs_edge(UndirectedEdge const &) const;

private:
  maybe_owned_ref<IUndirectedGraphView const> lhs;
  maybe_owned_ref<IUndirectedGraphView const> rhs;
  JoinedNodeView joined_nodes;
};

struct JoinedDigraphView : public IDiGraphView {
public:
  JoinedDigraphView() = delete;
  explicit JoinedDigraphView(maybe_owned_ref<IDiGraphView const> lhs,
                             maybe_owned_ref<IDiGraphView const> rhs);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  JoinedNodeView const &joined_nodes_view() const;

private:
  DirectedEdge fix_lhs_edge(DirectedEdge const &) const;
  DirectedEdge fix_rhs_edge(DirectedEdge const &) const;

private:
  maybe_owned_ref<IDiGraphView const> lhs;
  maybe_owned_ref<IDiGraphView const> rhs;
  JoinedNodeView joined_nodes;
};

struct JoinedMultiDigraphView : public IMultiDiGraphView {
public:
  JoinedMultiDigraphView() = delete;
  JoinedMultiDigraphView(maybe_owned_ref<IMultiDiGraphView const> lhs,
                         maybe_owned_ref<IMultiDiGraphView const> rhs);

  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  JoinedNodeView const &joined_nodes_view() const;

private:
  MultiDiEdge fix_lhs_edge(MultiDiEdge const &) const;
  MultiDiEdge fix_rhs_edge(MultiDiEdge const &) const;

private:
  maybe_owned_ref<IMultiDiGraphView const> lhs;
  maybe_owned_ref<IMultiDiGraphView const> rhs;
  JoinedNodeView joined_nodes;
};

struct AddDirectedEdgesView : public IDiGraphView {
public:
  AddDirectedEdgesView() = delete;

  explicit AddDirectedEdgesView(maybe_owned_ref<IDiGraphView const> g,
                                std::unordered_set<DirectedEdge> const &edges);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  maybe_owned_ref<IDiGraphView const> g;
  std::unordered_set<DirectedEdge> edges;
};

struct SingleSourceNodeView : public IDiGraphView {
public:
  SingleSourceNodeView() = delete;

  template <template <typename> class Ptr>
  explicit SingleSourceNodeView(Ptr<IDiGraphView const> g) : g(g) {}

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  maybe_owned_ref<DiGraphView const> g;
  tl::optional<AdjacencyDiGraph> singleton_src;
  tl::optional<JoinedDigraphView> joined_view;
  std::unique_ptr<AddDirectedEdgesView> added_edges_view;
};

struct ContractNodeView : public IDiGraphView {
  ContractNodeView() = delete;
  explicit ContractNodeView(IDiGraphView const &,
                            Node const &removed,
                            Node const &into);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  DirectedEdge fix_edge(DirectedEdge const &) const;

private:
  IDiGraphView const &g;
  Node from, to;
};

struct DiGraphViewStack : public IDiGraphView {
public:
  DiGraphViewStack() = default;

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  void add_view(
      std::function<std::unique_ptr<IDiGraphView>(IDiGraphView const &)> const
          &);

private:
  std::vector<std::unique_ptr<IDiGraphView>> views;
};

struct OpenMultiDiSubgraphView : public IOpenMultiDiGraphView {
public:
  OpenMultiDiSubgraphView() = delete;
  explicit OpenMultiDiSubgraphView(IOpenMultiDiGraphView const &,
                                   std::unordered_set<Node> const &);

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

private:
  IOpenMultiDiGraphView const &g;
};

DirectedEdge flipped(DirectedEdge const &);

DiGraphView unsafe_view_as_flipped(DiGraphView const &);
DiGraphView view_as_flipped(DiGraphView const &);

UndirectedGraphView unsafe_view_subgraph(UndirectedGraphView const &,
                                         std::unordered_set<Node> const &);
UndirectedGraphView view_subgraph(UndirectedGraphView const &,
                                  std::unordered_set<Node> const &);

DiGraphView unsafe_view_subgraph(DiGraphView const &,
                                 std::unordered_set<Node> const &);
DiGraphView view_subgraph(DiGraphView const &,
                          std::unordered_set<Node> const &);

MultiDiGraphView unsafe_view_subgraph(MultiDiGraphView const &,
                                      std::unordered_set<Node> const &);
MultiDiGraphView view_subgraph(MultiDiGraphView const &,
                               std::unordered_set<Node> const &);

OpenMultiDiGraphView unsafe_view_as_subgraph(OpenMultiDiGraphView const &,
                                             std::unordered_set<Node> const &);
OpenMultiDiGraphView view_subgraph(OpenMultiDiGraphView const &,
                                   std::unordered_set<Node> const &);

UndirectedGraphView unsafe_view_as_joined(UndirectedGraphView const &,
                                          UndirectedGraphView const &);
UndirectedGraphView view_as_joined(UndirectedGraphView const &,
                                   UndirectedGraphView const &);

DiGraphView unsafe_view_as_joined(DiGraphView const &, DiGraphView const &);
DiGraphView view_as_joined(DiGraphView const &, DiGraphView const &);

MultiDiGraphView unsafe_view_as_joined(MultiDiGraphView const &,
                                       MultiDiGraphView const &);
MultiDiGraphView view_as_joined(MultiDiGraphView const &,
                                MultiDiGraphView const &);

DiGraphView
    unsafe_view_with_added_edges(DiGraphView const &,
                                 std::unordered_set<DirectedEdge> const &);
DiGraphView view_with_added_edges(DiGraphView const &,
                                  std::unordered_set<DirectedEdge> const &);

DiGraphView unsafe_view_as_contracted(DiGraphView const &,
                                      Node const &from,
                                      Node const &into);
DiGraphView
    view_as_contracted(DiGraphView const &, Node const &from, Node const &into);

DiGraphView unsafe_view_as_contracted(DiGraphView const &,
                                      std::unordered_map<Node, Node> const &);
DiGraphView view_as_contracted(DiGraphView const &,
                               std::unordered_map<Node, Node> const &);

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

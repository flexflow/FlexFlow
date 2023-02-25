#ifndef _FLEXFLOW_UTILS_GRAPH_VIEWS_H
#define _FLEXFLOW_UTILS_GRAPH_VIEWS_H

#include "digraph.h"
#include "multidigraph.h"
#include "undirected.h"
#include "utils/bidict.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace utils {

struct FlippedView : public IDiGraphView {
public:
  FlippedView() = delete;
  explicit FlippedView(IDiGraphView const &);

  std::unordered_set<DirectedEdge> query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IDiGraphView const &g;
};

struct DiSubgraphView : public IDiGraphView {
public:
  DiSubgraphView() = delete;
  explicit DiSubgraphView(IDiGraphView const &, std::unordered_set<Node> const &);

  std::unordered_set<DirectedEdge> query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IDiGraphView const &g;
  std::unordered_set<Node> subgraph_nodes;
};

struct MultiDiSubgraphView : public IMultiDiGraphView {
public: 
  MultiDiSubgraphView() = delete;
  explicit MultiDiSubgraphView(IMultiDiGraphView const &, std::unordered_set<Node> const &);

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IMultiDiGraphView const &g;
  std::unordered_set<Node> subgraph_nodes;
};

struct NodeSource {
public:
  NodeSource() = default;

  Node fresh_node();
private:
  std::size_t next_node_idx = 0;
};

struct JoinNodeKey {
  JoinNodeKey() = delete;
  JoinNodeKey(Node const &, bool);

  bool operator==(JoinNodeKey const &) const;
  bool operator<(JoinNodeKey const &) const;

  Node node;
  bool is_left;
};

}
}

namespace std {
template <>
struct hash<::FlexFlow::utils::JoinNodeKey> {
  std::size_t operator()(::FlexFlow::utils::JoinNodeKey const &) const;
};
}

namespace FlexFlow {
namespace utils {

struct JoinedNodeView {
public:
  JoinedNodeView() = delete;
  explicit JoinedNodeView(IGraphView const &lhs, IGraphView const &rhs);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::pair<std::unordered_set<Node>, std::unordered_set<Node>> trace_nodes(std::unordered_set<Node> const &) const;

  Node at_join_key(JoinNodeKey const &) const;
  JoinNodeKey at_node(Node const &) const;
private:
  bidict<JoinNodeKey, Node> mapping;
  NodeSource node_source;
};

struct JoinedUndirectedGraphView : public IUndirectedGraphView {
public:
  JoinedUndirectedGraphView() = delete;
  explicit JoinedUndirectedGraphView(IUndirectedGraphView const &lhs, IUndirectedGraphView const &rhs);

  std::unordered_set<UndirectedEdge> query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  UndirectedEdge fix_lhs_edge(UndirectedEdge const &) const;
  UndirectedEdge fix_rhs_edge(UndirectedEdge const &) const;
private:
  IUndirectedGraphView const &lhs;
  IUndirectedGraphView const &rhs;
  JoinedNodeView joined_nodes;
};

struct JoinedDigraphView : public IDiGraphView {
public:
  JoinedDigraphView() = delete;
  explicit JoinedDigraphView(IDiGraphView const &lhs, IDiGraphView const &rhs);

  std::unordered_set<DirectedEdge> query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  DirectedEdge fix_lhs_edge(DirectedEdge const &) const;
  DirectedEdge fix_rhs_edge(DirectedEdge const &) const;
private:
  IDiGraphView const &lhs;
  IDiGraphView const &rhs;
  JoinedNodeView joined_nodes;
};

struct JoinedMultiDigraphView : public IMultiDiGraphView {
public:
  JoinedMultiDigraphView() = delete;
  explicit JoinedMultiDigraphView(IMultiDiGraphView const &lhs, IMultiDiGraphView const &rhs);

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  bidict<std::pair<Node, bool>, Node> node_mapping() const;
private:
  MultiDiEdge fix_lhs_edge(MultiDiEdge const &) const;
  MultiDiEdge fix_rhs_edge(MultiDiEdge const &) const;
private:
  IMultiDiGraphView const &lhs;
  IMultiDiGraphView const &rhs;
  JoinedNodeView joined_nodes;
};

DirectedEdge flipped(DirectedEdge const &);
FlippedView unsafe_view_as_flipped(IDiGraphView const &);
DiSubgraphView unsafe_view_subgraph(IDiGraphView const &, std::unordered_set<Node> const &);
MultiDiSubgraphView unsafe_view_subgraph(IMultiDiGraphView const &, std::unordered_set<Node> const &);
JoinedUndirectedGraphView unsafe_view_as_joined(IUndirectedGraphView const &, IUndirectedGraphView const &);
JoinedDigraphView unsafe_view_as_joined(IDiGraphView const &, IDiGraphView const &);
JoinedMultiDigraphView unsafe_view_as_joined(IMultiDiGraphView const &, IMultiDiGraphView const &);

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

}
}

VISITABLE_STRUCT(::FlexFlow::utils::JoinNodeKey, node, is_left);

#endif 

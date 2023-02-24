#ifndef _FLEXFLOW_UTILS_GRAPH_VIEWS_H
#define _FLEXFLOW_UTILS_GRAPH_VIEWS_H

#include "digraph.h"
#include "multidigraph.h"
#include "undirected.h"

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

struct JoinedUndirectedGraphView : public IUndirectedGraphView {
public:
  JoinedUndirectedGraphView() = delete;
  explicit JoinedUndirectedGraphView(IUndirectedGraphView const &lhs, IUndirectedGraphView const &rhs);

  std::unordered_set<UndirectedEdge> query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IUndirectedGraphView const &lhs;
  IUndirectedGraphView const &rhs;
};

struct JoinedDigraphView : public IDiGraphView {
public:
  JoinedDigraphView() = delete;
  explicit JoinedDigraphView(IDiGraphView const &lhs, IDiGraphView const &rhs);

  std::unordered_set<DirectedEdge> query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IDiGraphView const &lhs;
  IDiGraphView const &rhs;
};

struct JoinedMultiDigraphView : public IMultiDiGraphView {
public:
  JoinedMultiDigraphView() = delete;
  explicit JoinedMultiDigraphView(IMultiDiGraphView const &lhs, IMultiDiGraphView const &rhs);

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IMultiDiGraphView const &lhs;
  IMultiDiGraphView const &rhs;
};

struct UndirectedPermutationView : public IUndirectedGraphView {

};

struct DirectedPermutationView : public IDiGraphView {

};

struct MultiDiPermutationView : public IMultiDiGraphView {

};

DirectedEdge flipped(DirectedEdge const &);
FlippedView unsafe_view_as_flipped(IDiGraphView const &);
DiSubgraphView unsafe_view_subgraph(IDiGraphView const &, std::unordered_set<Node> const &);
MultiDiSubgraphView unsafe_view_subgraph(IMultiDiGraphView const &, std::unordered_set<Node> const &);
JoinedUndirectedGraphView unsafe_view_as_joined(IUndirectedGraphView const &, IUndirectedGraphView const &);
JoinedDigraphView unsafe_view_as_joined(IDiGraphView const &, IDiGraphView const &);
JoinedMultiDigraphView unsafe_view_as_joined(IMultiDiGraphView const &, IMultiDiGraphView const &);


}
}

#endif 

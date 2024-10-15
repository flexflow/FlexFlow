#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_VIEWS_VIEWS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_VIEWS_VIEWS_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/undirected/undirected_graph_view.h"
#include "utils/graph/views/join_node_key.dtg.h"

namespace FlexFlow {

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

UndirectedGraphView view_subgraph(UndirectedGraphView const &,
                                  std::unordered_set<Node> const &);

DiGraphView view_subgraph(DiGraphView const &,
                          std::unordered_set<Node> const &);

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

} // namespace FlexFlow

#endif

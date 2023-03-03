#ifndef _FLEXFLOW_UTILS_GRAPH_CONVERSION_H
#define _FLEXFLOW_UTILS_GRAPH_CONVERSION_H

#include "multidigraph.h"
#include "digraph.h"
#include "undirected.h"
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "mpark/variant.hpp"
#include "open_graphs.h"

namespace FlexFlow {

UndirectedEdge to_undirected_edge(DirectedEdge const &);
std::unordered_set<UndirectedEdge> to_undirected_edges(std::unordered_set<DirectedEdge> const &);
UndirectedEdge to_undirected_edge(MultiDiEdge const &);
std::unordered_set<UndirectedEdge> to_undirected_edges(std::unordered_set<MultiDiEdge> const &);

std::unordered_set<DirectedEdge> to_directed_edges(UndirectedEdge const &);
std::unordered_set<DirectedEdge> to_directed_edges(std::unordered_set<UndirectedEdge> const &);
DirectedEdge to_directed_edge(MultiDiEdge const &);
std::unordered_set<DirectedEdge> to_directed_edges(std::unordered_set<MultiDiEdge> const &);

std::unordered_set<MultiDiEdge> to_multidigraph_edges(UndirectedEdge const &);
std::unordered_set<MultiDiEdge> to_multidigraph_edges(std::unordered_set<UndirectedEdge> const &);
MultiDiEdge to_multidigraph_edge(DirectedEdge const &);
std::unordered_set<MultiDiEdge> to_multidigraph_edges(std::unordered_set<DirectedEdge> const &);

template <typename Undirected>
Undirected to_undirected(IDiGraphView const &directed) {
  static_assert(std::is_base_of<IUndirectedGraph, Undirected>::value, "Error");
  Undirected undirected;
  DirectedEdgeQuery edge_query_all;
  NodeQuery node_query_all;
  
  std::unordered_set<DirectedEdge> directed_edges = directed.query_edges(edge_query_all);
  std::unordered_set<Node> directed_nodes = directed.query_nodes(node_query_all);

  auto directed_node_to_undirected_node = [&]() -> std::unordered_map<Node, Node> {
    std::unordered_map<Node, Node> result;
    for (Node const &directed_node : directed_nodes) {
      result[directed_node] = undirected.add_node();
    }
    return result;
  };

  for (DirectedEdge const &directed_edge : directed_edges) {
    undirected.add_edge(to_undirected_edge(directed_edge));
  }

  return undirected;
}

struct ViewDiGraphAsUndirectedGraph : public IUndirectedGraphView {
public:
  ViewDiGraphAsUndirectedGraph() = delete;
  explicit ViewDiGraphAsUndirectedGraph(IDiGraphView const &);
  explicit ViewDiGraphAsUndirectedGraph(std::shared_ptr<IDiGraphView> const &);

  std::unordered_set<UndirectedEdge> query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  std::shared_ptr<IDiGraphView> const shared;
  IDiGraphView const *directed;
};

struct ViewDiGraphAsMultiDiGraph : public IMultiDiGraphView {
public:
  ViewDiGraphAsMultiDiGraph() = delete;
  explicit ViewDiGraphAsMultiDiGraph(IDiGraphView const &);
  explicit ViewDiGraphAsMultiDiGraph(std::shared_ptr<IDiGraphView> const &);

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  std::shared_ptr<IDiGraphView> const shared = nullptr;
  IDiGraphView const *directed;
};

struct ViewMultiDiGraphAsDiGraph : public IDiGraphView {
public:
  ViewMultiDiGraphAsDiGraph() = delete;
  explicit ViewMultiDiGraphAsDiGraph(IMultiDiGraphView const &);
  explicit ViewMultiDiGraphAsDiGraph(std::shared_ptr<IMultiDiGraphView> const &);

  std::unordered_set<DirectedEdge> query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  std::shared_ptr<IMultiDiGraphView> const shared;
  IMultiDiGraphView const *multidi;
};

struct ViewOpenMultiDiGraphAsMultiDiGraph : public IMultiDiGraphView { 
public:
  ViewOpenMultiDiGraphAsMultiDiGraph() = delete;
  explicit ViewOpenMultiDiGraphAsMultiDiGraph(IOpenMultiDiGraph const &);

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IOpenMultiDiGraphView const &g;
};


std::unique_ptr<IUndirectedGraphView> unsafe_view_as_undirected(IDiGraphView const &);
std::unique_ptr<IUndirectedGraphView> view_as_undirected(std::shared_ptr<IDiGraphView const>);

std::unique_ptr<IMultiDiGraphView> unsafe_view_as_multidigraph(IDiGraphView const &);
std::unique_ptr<IMultiDiGraphView> view_as_multidigraph(std::shared_ptr<IDiGraphView const>);

std::unique_ptr<IDiGraphView> unsafe_view_as_digraph(IMultiDiGraphView const &);
std::unique_ptr<IDiGraphView> view_as_digraph(std::shared_ptr<IMultiDiGraphView const>);

std::unique_ptr<IMultiDiGraphView> unsafe_view_as_multidigraph(std::shared_ptr<IOpenMultiDiGraphView const>);
std::unique_ptr<IMultiDiGraphView> view_as_multidigraph(std::shared_ptr<IOpenMultiDiGraphView const>);
}

#endif

#ifndef _FLEXFLOW_UTILS_GRAPH_CONVERSION_H
#define _FLEXFLOW_UTILS_GRAPH_CONVERSION_H

#include "multidigraph.h"
#include "digraph.h"
#include "undirected.h"
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "mpark/variant.hpp"

namespace FlexFlow {
namespace utils {
namespace graph {

undirected::Edge to_undirected_edge(digraph::Edge const &);
undirected::Edge to_undirected_edge(multidigraph::Edge const &);
digraph::Edge to_directed_edge(undirected::Edge const &);
digraph::Edge to_directed_edge(multidigraph::Edge const &);
multidigraph::Edge to_multidigraph_edge(undirected::Edge const &);
multidigraph::Edge to_multidigraph_edge(digraph::Edge const &);

template <typename Undirected>
Undirected to_undirected(IDiGraph const &directed) {
  static_assert(std::is_base_of<IUndirectedGraph, Undirected>::value, "Error");
  Undirected undirected;
  digraph::EdgeQuery edge_query_all;
  NodeQuery node_query_all;
  
  std::unordered_set<digraph::Edge> directed_edges = directed.query_edges(edge_query_all);
  std::unordered_set<Node> directed_nodes = directed.query_nodes(node_query_all);

  auto directed_node_to_undirected_node = [&]() -> std::unordered_map<Node, Node> {
    std::unordered_map<Node, Node> result;
    for (Node const &directed_node : directed_nodes) {
      result[directed_node] = undirected.add_node();
    }
    return result;
  };

  for (digraph::Edge const &directed_edge : directed_edges) {
    undirected.add_edge(to_undirected_edge(directed_edge));
  }

  return undirected;
}

struct ViewDiGraphAsUndirectedGraph : public IUndirectedGraphView {
public:
  explicit ViewDiGraphAsUndirectedGraph(IDiGraphView const &);
  explicit ViewDiGraphAsUndirectedGraph(std::shared_ptr<IDiGraphView> const &);

  std::unordered_set<undirected::Edge> query_edges(undirected::EdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  std::shared_ptr<IDiGraphView> directed;
};

struct ViewDiGraphAsMultiDiGraph : public IMultiDiGraphView {
public:
  explicit ViewDiGraphAsMultiDiGraph(IDiGraphView const &);
  explicit ViewDiGraphAsMultiDiGraph(std::shared_ptr<IDiGraphView> const &);

  std::unordered_set<multidigraph::Edge> query_edges(multidigraph::EdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  std::shared_ptr<IDiGraphView> directed;
};

struct ViewMultiDiGraphAsDiGraph : public IDiGraphView {
public:
  explicit ViewMultiDiGraphAsDiGraph(IMultiDiGraphView const &);
  explicit ViewMultiDiGraphAsDiGraph(std::shared_ptr<IMultiDiGraphView> const &);

  std::unordered_set<digraph::Edge> query_edges(digraph::EdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  mpark::variant<std::shared_ptr<IDiGraphView>, IDiGraphView const *> directed;
};

ViewDiGraphAsUndirectedGraph unsafe_view_as_undirected(IDiGraphView const &directed);
ViewDiGraphAsUndirectedGraph view_as_undirected(std::shared_ptr<IDiGraph> const &directed);
ViewDiGraphAsMultiDiGraph unsafe_view_as_multidigraph(IDiGraphView const &directed);
ViewDiGraphAsMultiDiGraph view_as_multidigraph(std::shared_ptr<IDiGraph> const &directed);
ViewMultiDiGraphAsDiGraph unsafe_view_as_digraph(IMultiDiGraph const &multidigraph);
ViewMultiDiGraphAsDiGraph view_as_digraph(std::shared_ptr<IMultiDiGraph> const &multidigraph);

}
}
}

#endif

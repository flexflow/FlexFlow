#ifndef _FLEXFLOW_UTILS_GRAPH_CONVERSION_H
#define _FLEXFLOW_UTILS_GRAPH_CONVERSION_H

#include "multidigraph.h"
#include "digraph.h"
#include "undirected.h"
#include <memory>
#include <type_traits>
#include <unordered_map>

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

struct UndirectedGraphViewFromDirected : public IUndirectedGraphView {
public:
  explicit UndirectedGraphViewFromDirected(std::shared_ptr<IDiGraphView> const &);

  std::unordered_set<undirected::Edge> query_edges(undirected::EdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  std::shared_ptr<IDiGraphView> directed;
};

UndirectedGraphViewFromDirected view_undirected(std::shared_ptr<IDiGraph> const &directed);

}
}
}

#endif

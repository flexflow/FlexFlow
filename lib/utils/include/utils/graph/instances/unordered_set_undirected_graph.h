#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_UNDIRECTED_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_UNDIRECTED_GRAPH_H

#include "utils/graph/node/node_source.h"
#include "utils/graph/undirected/i_undirected_graph.h"

namespace FlexFlow {

struct UnorderedSetUndirectedGraph final : public IUndirectedGraph {
public:
  UnorderedSetUndirectedGraph();

  Node add_node() override;
  void add_node_unsafe(Node const &) override;
  void remove_node_unsafe(Node const &) override;
  void add_edge(UndirectedEdge const &) override;
  void remove_edge(UndirectedEdge const &) override;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
  std::unordered_set<UndirectedEdge> query_edges(UndirectedEdgeQuery const &) const override;

  UnorderedSetUndirectedGraph *clone() const override;
private:
  UnorderedSetUndirectedGraph(NodeSource const &,
                              std::unordered_set<Node> const &,
                              std::unordered_set<UndirectedEdge> const &);

  NodeSource node_source;
  std::unordered_set<Node> nodes;
  std::unordered_set<UndirectedEdge> edges;
};

} // namespace FlexFlow

#endif

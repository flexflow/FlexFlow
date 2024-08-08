#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_ADJACENCY_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_ADJACENCY_MULTIDIGRAPH_H

#include "utils/graph/multidigraph/multidiedge_source.h"
#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/graph/node/node_source.h"

namespace FlexFlow {

struct AdjacencyMultiDiGraph final : public IMultiDiGraph {
public:
  AdjacencyMultiDiGraph();

  Node add_node() override;
  MultiDiEdge add_edge(Node const &, Node const &) override;
  void remove_node(Node const &) override;
  void remove_edge(MultiDiEdge const &) override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &) const override;
  Node get_multidiedge_src(MultiDiEdge const &) const override;
  Node get_multidiedge_dst(MultiDiEdge const &) const override;
  void inplace_materialize_from(MultiDiGraphView const &) override;

  AdjacencyMultiDiGraph *clone() const override;

private:
  AdjacencyMultiDiGraph(
      NodeSource const &,
      MultiDiEdgeSource const &,
      std::unordered_map<
          Node,
          std::unordered_map<Node, std::unordered_set<MultiDiEdge>>> const &,
      std::unordered_map<MultiDiEdge, std::pair<Node, Node>> const &);

private:
  NodeSource node_source;
  MultiDiEdgeSource edge_source;
  std::unordered_map<Node,
                     std::unordered_map<Node, std::unordered_set<MultiDiEdge>>>
      adjacency;
  std::unordered_map<MultiDiEdge, std::pair<Node, Node>> edge_nodes;
};

} // namespace FlexFlow

#endif

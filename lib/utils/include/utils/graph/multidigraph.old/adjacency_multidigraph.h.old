#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_ADJACENCY_MULTIDIGRAPH
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_ADJACENCY_MULTIDIGRAPH

#include "utils/graph/multidigraph/i_multidigraph.h"

namespace FlexFlow {

class AdjacencyOpenMultiDiGraph;

class AdjacencyMultiDiGraph : virtual public IMultiDiGraph {
public:
  AdjacencyMultiDiGraph() = default;
  Node add_node() override;
  void add_node_unsafe(Node const &) override;
  NodePort add_node_port() override;
  void add_node_port_unsafe(NodePort const &) override;
  void remove_node_unsafe(Node const &) override;
  void add_edge(Edge const &) override;
  void remove_edge(Edge const &) override;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  AdjacencyMultiDiGraph *clone() const override;

  ~AdjacencyMultiDiGraph() = default;

private:
  using ContentsType = std::unordered_map<
      Node,
      std::unordered_map<
          Node,
          std::unordered_map<NodePort, std::unordered_set<NodePort>>>>;

  AdjacencyMultiDiGraph(std::size_t next_node_idx,
                        std::size_t next_node_port,
                        ContentsType const &adjacency);

private:
  std::size_t next_node_idx = 0;
  std::size_t next_node_port = 0;
  ContentsType adjacency;

  friend AdjacencyOpenMultiDiGraph;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(AdjacencyMultiDiGraph);

} // namespace FlexFlow

#endif

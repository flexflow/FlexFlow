#ifndef _FLEXFLOW_UTILS_ADJACENCY_MULTIGRAPH_H
#define _FLEXFLOW_UTILS_ADJACENCY_MULTIGRAPH_H

#include "multidigraph.h"
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

class AdjacencyMultiDiGraph : public IMultiDiGraph {
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

  AdjacencyMultiDiGraph *clone() const override {
    return new AdjacencyMultiDiGraph(this->next_node_idx, this->adjacency);
  }

private:
  using ContentsType = std::unordered_map<
      Node,
      std::unordered_map<
          Node,
          std::unordered_map<NodePort, std::unordered_set<NodePort>>>>;

  AdjacencyMultiDiGraph(std::size_t next_node_idx,  ContentsType const & adjacency, std::size_t next_node_port=0)
      : next_node_idx(next_node_idx), next_node_port(next_node_port), adjacency(adjacency) {}
      
  std::size_t next_node_idx = 0;
  std::size_t next_node_port = 0;
  ContentsType adjacency;
};

static_assert(is_rc_copy_virtual_compliant<AdjacencyMultiDiGraph>::value,
              RC_COPY_VIRTUAL_MSG);

} // namespace FlexFlow

#endif

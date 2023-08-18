#ifndef _FLEXFLOW_UTILS_GRAPH_ADJACENCY_DIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_ADJACENCY_DIGRAPH_H

#include <unordered_map>
#include <unordered_set>
#include "digraph_interfaces.h" 

namespace FlexFlow {

class AdjacencyDiGraph : public IDiGraph {
public:
  void add_node(Node const &) override;
  void add_node_unsafe(Node const &) override;
  void remove_node_unsafe(Node const &) override;
  void add_edge(Edge const &) override;
  void remove_edge(Edge const &) override;
  std::unordered_set<Edge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  bool operator==(AdjacencyDiGraph const &) const;
  bool operator!=(AdjacencyDiGraph const &) const;

  AdjacencyDiGraph *clone() const override {
    return new AdjacencyDiGraph(this->next_node_idx, this->adjacency);
  }

private:
  using ContentsType = std::unordered_map<Node, std::unordered_set<Node>>;

  AdjacencyDiGraph(std::size_t, ContentsType);

  std::size_t next_node_idx = 0;
  ContentsType adjacency;
};

static_assert(is_rc_copy_virtual_compliant<AdjacencyDiGraph>::value,
              RC_COPY_VIRTUAL_MSG);

} // namespace FlexFlow

#endif

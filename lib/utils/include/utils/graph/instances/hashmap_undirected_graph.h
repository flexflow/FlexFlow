#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_HASHMAP_UNDIRECTED_GRAPH_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_HASHMAP_UNDIRECTED_GRAPH_H

#include "utils/graph/undirected/i_undirected_graph.h"

namespace FlexFlow {

class HashmapUndirectedGraph : public IUndirectedGraph {
public:
  HashmapUndirectedGraph() = default;

  Node add_node() override;
  void add_node_unsafe(Node const &) override;
  void remove_node_unsafe(Node const &) override;
  void add_edge(Edge const &) override;
  void remove_edge(Edge const &) override;
  std::unordered_set<Edge>
      query_edges(UndirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  friend bool operator==(HashmapUndirectedGraph const &,
                         HashmapUndirectedGraph const &);
  friend bool operator!=(HashmapUndirectedGraph const &,
                         HashmapUndirectedGraph const &);

  HashmapUndirectedGraph *clone() const override {
    return new HashmapUndirectedGraph(this->next_node_idx, this->adjacency);
  }

private:
  using ContentsType = std::unordered_map<Node, std::unordered_set<Node>>;

  HashmapUndirectedGraph(std::size_t next_node_idx, ContentsType adjacency)
      : next_node_idx(next_node_idx), adjacency(adjacency) {}
  std::size_t next_node_idx = 0;
  ContentsType adjacency;
};

} // namespace FlexFlow

#endif

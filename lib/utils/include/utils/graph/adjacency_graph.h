#ifndef _FLEXFLOW_UTILS_ADJACENCY_GRAPH_H
#define _FLEXFLOW_UTILS_ADJACENCY_GRAPH_H 

#include "graph_core.h"

namespace FlexFlow {
namespace utils {

class AdjacencyMultiDiGraph : public IMultiDiGraph {
public:
  Node add_node() override;
  void add_edge(Edge const &) override;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const override;
  std::unordered_set<std::size_t> query_nodes(NodeQuery const &) const override;

private:
  std::size_t next_node_idx = 0;
  std::unordered_map<Node, 
    std::unordered_map<Node,
      std::unordered_map<std::size_t, std::unordered_set<std::size_t>>>> adjacency;
};

}
}

#endif

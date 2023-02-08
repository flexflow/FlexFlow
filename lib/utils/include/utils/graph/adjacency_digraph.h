#ifndef _FLEXFLOW_UTILS_GRAPH_ADJACENCY_DIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_ADJACENCY_DIGRAPH_H

#include "digraph.h"
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {
namespace utils {
namespace graph {
namespace digraph {

class AdjacencyDiGraph : public IDiGraph {
public:
  Node add_node() override;
  void add_edge(Edge const &) override;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  std::size_t next_node_idx = 0;
  std::unordered_map<Node, std::unordered_set<Node>> adjacency;
};

}
}
}
}

#endif

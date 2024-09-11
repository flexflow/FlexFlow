#include "utils/graph/open_dataflow_graph/algorithms/get_source_nodes.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_incoming_edges.h"

namespace FlexFlow {

std::unordered_set<Node> get_source_nodes(OpenDataflowGraphView const &g) {
  auto is_source_node = [&](Node const &n) {
    std::vector<OpenDataflowEdge> incoming_edges = get_incoming_edges(g, n);
    return incoming_edges.empty();
  };

  return filter(get_nodes(g), is_source_node);
}

} // namespace FlexFlow

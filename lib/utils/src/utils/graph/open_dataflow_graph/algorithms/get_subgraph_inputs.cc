#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_inputs.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/overload.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowValue> get_subgraph_inputs(OpenDataflowGraphView const &g,
                                                          std::unordered_set<Node> const &subgraph_nodes) {
  std::unordered_set<OpenDataflowEdge> relevant_edges;
  for (std::vector<OpenDataflowEdge> const &incoming : values(get_incoming_edges(g, subgraph_nodes))) {
    extend(relevant_edges, incoming);
  }

  return transform(relevant_edges, get_open_dataflow_edge_source);
}

} // namespace FlexFlow

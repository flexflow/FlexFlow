#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_inputs.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/overload.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowValue>
    get_subgraph_inputs(OpenDataflowGraphView const &g,
                        std::unordered_set<Node> const &subgraph_nodes) {
  std::unordered_set<OpenDataflowEdge> relevant_edges;
  for (std::vector<OpenDataflowEdge> const &incoming :
       values(get_incoming_edges(g, subgraph_nodes))) {
    auto comes_from_outside_subgraph = [&](OpenDataflowEdge const &e) -> bool {
      return e.visit<bool>(overload{
          [](DataflowInputEdge const &) { return true; },
          [&](DataflowEdge const &ee) {
            assert(contains(subgraph_nodes, ee.dst.node));
            return !contains(subgraph_nodes, ee.src.node);
          },
      });
    };

    extend(relevant_edges, filter(incoming, comes_from_outside_subgraph));
  }

  return transform(relevant_edges, get_open_dataflow_edge_source);
}

} // namespace FlexFlow

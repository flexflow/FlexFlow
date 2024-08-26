#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"
#include "utils/containers/transform.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_incoming_edges.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"

namespace FlexFlow {

std::unordered_set<DataflowGraphInput>
    get_inputs(OpenDataflowGraphView const &g) {
  return g.get_inputs();
}

std::vector<OpenDataflowValue> get_inputs(OpenDataflowGraphView const &g,
                                          Node const &n) {
  return transform(get_incoming_edges(g, n), [](OpenDataflowEdge const &e) {
    return get_open_dataflow_edge_source(e);
  });
}

} // namespace FlexFlow

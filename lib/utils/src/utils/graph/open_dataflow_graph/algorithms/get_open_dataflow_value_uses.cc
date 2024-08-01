#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_value_uses.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::unordered_set<DataflowInput> get_open_dataflow_value_uses(OpenDataflowGraphView const &view,
                                                               OpenDataflowValue const &value) {
  std::unordered_set<OpenDataflowEdge> edges = view.query_edges(open_dataflow_input_edge_query_all_outgoing_from(value));

  return transform(edges, get_open_dataflow_edge_dst);
}

} // namespace FlexFlow

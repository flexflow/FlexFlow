#include "utils/graph/open_dataflow_graph/algorithms/get_incoming_edge.h"
#include "utils/containers/get_only.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.h"

namespace FlexFlow {

OpenDataflowEdge get_incoming_edge(OpenDataflowGraphView const &g,
                                   DataflowInput const &i) {
  OpenDataflowEdgeQuery query = open_dataflow_edge_query_all_incoming_to(i);
  std::unordered_set<OpenDataflowEdge> query_result = g.query_edges(query);

  return get_only(query_result);
}

} // namespace FlexFlow

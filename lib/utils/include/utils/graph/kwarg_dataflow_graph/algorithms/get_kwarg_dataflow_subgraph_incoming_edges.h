#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_SUBGRAPH_INCOMING_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_SUBGRAPH_INCOMING_EDGES_H

#include "utils/containers/set_minus.h"
#include "utils/containers/set_of.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

template <typename SlotName>
std::set<KwargDataflowEdge<SlotName>>
    get_kwarg_dataflow_subgraph_incoming_edges(
        KwargDataflowGraphView<SlotName> const &g,
        std::set<Node> const &subgraph) {

  KwargDataflowEdgeQuery<SlotName> query = KwargDataflowEdgeQuery<SlotName>{
      /*src_nodes=*/query_set<Node>::match_except_values_in(subgraph),
      /*src_slots=*/query_set<SlotName>::matchall(),
      /*dst_nodes=*/query_set<Node>::match_values_in(subgraph),
      /*dst_slots=*/query_set<SlotName>::matchall(),
  };

  return g.query_edges(query);
}

} // namespace FlexFlow

#endif

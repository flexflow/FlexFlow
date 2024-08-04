#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_GRAPH_DATA_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/labelled_open_dataflow_graph_data.dtg.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_values.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
LabelledOpenDataflowGraphData<NodeLabel, ValueLabel> 
  get_graph_data(LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &g) {
  
  std::unordered_map<Node, NodeLabel> node_data = generate_map(get_nodes(g),
                                                               [&](Node const &n) { return g.at(n); });

  std::unordered_set<OpenDataflowEdge> edges = get_edges(g);

  std::unordered_set<DataflowGraphInput> inputs = g.get_inputs();

  std::unordered_map<OpenDataflowValue, ValueLabel> value_data = generate_map(get_open_dataflow_values(g),
                                                                              [&](OpenDataflowValue const &v) { return g.at(v); });

  return LabelledOpenDataflowGraphData<NodeLabel, ValueLabel>{
    node_data,
    edges,
    inputs,
    value_data,
  };
}

} // namespace FlexFlow

#endif

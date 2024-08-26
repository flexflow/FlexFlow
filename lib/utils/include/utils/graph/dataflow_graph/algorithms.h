#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_H

#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include <vector>

namespace FlexFlow {

std::unordered_set<DataflowEdge> get_edges(DataflowGraphView const &);
std::vector<DataflowOutput> get_inputs(DataflowGraphView const &, Node const &);
std::vector<DataflowInput> get_dataflow_inputs(DataflowGraphView const &,
                                               Node const &);
std::vector<DataflowOutput> get_outputs(DataflowGraphView const &,
                                        Node const &);
std::unordered_set<DataflowOutput>
    get_all_dataflow_outputs(DataflowGraphView const &);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_INPUTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_INPUTS_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

std::unordered_set<DataflowGraphInput>
    get_inputs(OpenDataflowGraphView const &);
std::vector<OpenDataflowValue> get_inputs(OpenDataflowGraphView const &,
                                          Node const &);

} // namespace FlexFlow

#endif

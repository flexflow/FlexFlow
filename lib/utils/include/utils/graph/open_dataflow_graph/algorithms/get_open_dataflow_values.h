#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_DATAFLOW_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_DATAFLOW_VALUES_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowValue>
    get_open_dataflow_values(OpenDataflowGraphView const &);

} // namespace FlexFlow

#endif

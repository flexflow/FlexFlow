#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_DATAFLOW_VALUE_USES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_DATAFLOW_VALUE_USES_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

std::unordered_set<DataflowInput>
    get_open_dataflow_value_uses(OpenDataflowGraphView const &view,
                                 OpenDataflowValue const &value);

} // namespace FlexFlow

#endif

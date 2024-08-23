#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_VALUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_VALUE_H

#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"
#include <optional>

namespace FlexFlow {

std::optional<DataflowOutput> try_get_dataflow_output(OpenDataflowValue const &);
std::optional<DataflowGraphInput> try_get_dataflow_graph_input(OpenDataflowValue const &);

} // namespace FlexFlow

#endif

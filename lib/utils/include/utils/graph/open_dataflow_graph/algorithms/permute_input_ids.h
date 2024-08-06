#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_INPUT_IDS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_INPUT_IDS_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/algorithms/new_dataflow_graph_input.dtg.h"

namespace FlexFlow {

OpenDataflowGraphView permute_input_ids(OpenDataflowGraphView const &,
                                        bidict<NewDataflowGraphInput, DataflowGraphInput> const &input_mapping);

} // namespace FlexFlow

#endif

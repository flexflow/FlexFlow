#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GENERATE_NEW_INPUT_ID_PERMUTATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GENERATE_NEW_INPUT_ID_PERMUTATION_H

#include "utils/graph/open_dataflow_graph/algorithms/new_dataflow_graph_input.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

bidict<NewDataflowGraphInput, DataflowGraphInput>
    generate_new_input_id_permutation(OpenDataflowGraphView const &);

} // namespace FlexFlow

#endif

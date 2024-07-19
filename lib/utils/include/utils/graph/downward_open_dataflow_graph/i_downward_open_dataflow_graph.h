#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_DATAFLOW_GRAPH_I_DOWNWARD_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_DATAFLOW_GRAPH_I_DOWNWARD_OPEN_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/dataflow_output.dtg.h"
#include "utils/graph/dataflow_graph/node_added_result.dtg.h"

namespace FlexFlow {

struct IDownwardOpenDataflowGraph
    : virtual public IDownwardOpenDataflowGraphView {
  virtual NodeAddedResult add_node(std::vector<DataflowOutput> const &inputs,
                                   int num_outputs) = 0;
};

} // namespace FlexFlow

#endif

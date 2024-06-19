#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_I_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_I_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/node_added_result.dtg.h"
#include "utils/graph/dataflow_graph/i_dataflow_graph_view.h"

namespace FlexFlow {

struct IDataflowGraph : virtual public IDataflowGraphView {
  virtual NodeAddedResult add_node(std::vector<DataflowOutput> const &inputs,
                                   int num_outputs) = 0;
  virtual IDataflowGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDataflowGraph);

} // namespace FlexFlow

#endif

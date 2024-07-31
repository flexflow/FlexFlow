#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_I_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_I_OPEN_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/node_added_result.dtg.h"
#include "utils/graph/open_dataflow_graph/i_open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

struct IOpenDataflowGraph : virtual public IOpenDataflowGraphView {
  virtual NodeAddedResult add_node(std::vector<OpenDataflowValue> const &inputs,
                                   int num_outputs) = 0;
  virtual DataflowGraphInput add_input() = 0;
  virtual IOpenDataflowGraph *clone() const = 0;

  virtual ~IOpenDataflowGraph() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenDataflowGraph);

} // namespace FlexFlow

#endif

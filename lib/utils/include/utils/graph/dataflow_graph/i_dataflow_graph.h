#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_I_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_I_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/dataflow_graph/i_dataflow_graph_view.h"
#include "utils/graph/dataflow_graph/node_added_result.dtg.h"

namespace FlexFlow {

struct IDataflowGraph : virtual public IDataflowGraphView {
  virtual NodeAddedResult add_node(std::vector<DataflowOutput> const &inputs,
                                   int num_outputs) = 0;

  virtual void add_node_unsafe(Node const &node,
                               std::vector<DataflowOutput> const &inputs,
                               std::vector<DataflowOutput> const &outputs) = 0;

  virtual void inplace_materialize_from(DataflowGraphView const &) = 0;

  virtual IDataflowGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDataflowGraph);

} // namespace FlexFlow

#endif

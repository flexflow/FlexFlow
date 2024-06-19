#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/dataflow_graph/node_added_result.dtg.h"
#include "utils/graph/dataflow_graph/i_dataflow_graph.h"

namespace FlexFlow {

struct DataflowGraph : virtual DataflowGraphView {
public:
  NodeAddedResult add_node(std::vector<DataflowOutput> const &inputs,
                           int num_outputs);
private:
  IDataflowGraph const &get_interface() const;
  IDataflowGraph &get_interface();
};

} // namespace FlexFlow

#endif

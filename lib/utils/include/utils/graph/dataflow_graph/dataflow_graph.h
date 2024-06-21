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
  
  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<DataflowEdge> query_edges(DataflowEdgeQuery const &) const;
  std::unordered_set<DataflowOutput> query_outputs(DataflowOutputQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IDataflowGraph, T>::value,
                                 DataflowGraph>::type
      create() {
    return DataflowGraph(make_cow_ptr<T>());
  }

protected:
  using DataflowGraphView::DataflowGraphView;

private:
  IDataflowGraph &get_interface();
  IDataflowGraph const &get_interface() const;

  friend struct GraphInternal;
};

} // namespace FlexFlow

#endif

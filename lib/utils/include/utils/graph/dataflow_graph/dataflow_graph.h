#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/dataflow_graph/i_dataflow_graph.h"
#include "utils/graph/dataflow_graph/node_added_result.dtg.h"

namespace FlexFlow {

struct DataflowGraph : virtual public DataflowGraphView {
public:
  NodeAddedResult add_node(std::vector<DataflowOutput> const &inputs,
                           int num_outputs);

  void add_node_unsafe(Node const &node,
                       std::vector<DataflowOutput> const &inputs,
                       std::vector<DataflowOutput> const &outputs);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<DataflowEdge> query_edges(DataflowEdgeQuery const &) const;
  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IDataflowGraph, T>::value,
                                 DataflowGraph>::type
      create() {
    return DataflowGraph(make_cow_ptr<T>());
  }

  template <typename T>
  static std::enable_if_t<std::is_base_of_v<IDataflowGraph, T>, DataflowGraph>
      create_copy_of(DataflowGraphView const &view) {
    cow_ptr_t<T> impl = make_cow_ptr<T>();
    impl.get_mutable()->inplace_materialize_from(view);
    return DataflowGraph(std::move(impl));
  }

protected:
  using DataflowGraphView::DataflowGraphView;

private:
  IDataflowGraph &get_interface();
  IDataflowGraph const &get_interface() const;
};

} // namespace FlexFlow

#endif

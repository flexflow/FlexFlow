#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/node_added_result.dtg.h"
#include "utils/graph/open_dataflow_graph/i_open_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

struct OpenDataflowGraph : virtual public OpenDataflowGraphView {
public:
  NodeAddedResult add_node(std::vector<OpenDataflowValue> const &inputs,
                           int num_outputs);
  DataflowGraphInput add_input();

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IOpenDataflowGraph, T>::value,
                                 OpenDataflowGraph>::type
      create(Args &&...args) {
    return OpenDataflowGraph(make_cow_ptr<T>(std::forward<Args>(args)...));
  }
protected:
  using OpenDataflowGraphView::OpenDataflowGraphView;

private:
  IOpenDataflowGraph &get_interface();
  IOpenDataflowGraph const &get_interface() const;
};

} // namespace FlexFlow

#endif

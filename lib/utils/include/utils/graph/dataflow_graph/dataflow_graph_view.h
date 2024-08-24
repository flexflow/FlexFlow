#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/dataflow_graph/dataflow_edge_query.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_output_query.dtg.h"
#include "utils/graph/dataflow_graph/i_dataflow_graph_view.h"
#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

struct DataflowGraphView : virtual public DiGraphView {
  DataflowGraphView(DataflowGraphView const &) = default;
  DataflowGraphView &operator=(DataflowGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<DataflowEdge> query_edges(DataflowEdgeQuery const &) const;
  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &) const;

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IDataflowGraphView, T>::value,
                                 DataflowGraphView>::type
      create(Args &&...args) {
    return DataflowGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using DiGraphView::DiGraphView;

private:
  IDataflowGraphView const &get_interface() const;
};

} // namespace FlexFlow

#endif

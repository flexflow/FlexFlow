#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_OPEN_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/i_open_dataflow_graph_view.h"

namespace FlexFlow {

struct OpenDataflowGraphView : virtual public DataflowGraphView {
public:
  OpenDataflowGraphView(OpenDataflowGraphView const &) = default;
  OpenDataflowGraphView &operator=(OpenDataflowGraphView const &) = default;

  std::unordered_set<DataflowGraphInput> get_inputs() const;
  std::unordered_set<OpenDataflowEdge>
      query_edges(OpenDataflowEdgeQuery const &) const;

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<IOpenDataflowGraphView, T>::value,
                              OpenDataflowGraphView>::type
      create(Args &&...args) {
    return OpenDataflowGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using DataflowGraphView::DataflowGraphView;

private:
  IOpenDataflowGraphView const &get_interface() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(OpenDataflowGraphView);

} // namespace FlexFlow

#endif

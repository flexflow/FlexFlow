#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_DATAFLOW_GRAPH_I_DOWNWARD_OPEN_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_DATAFLOW_GRAPH_I_DOWNWARD_OPEN_DATAFLOW_GRAPH_VIEW_H

namespace FlexFlow {

struct IDownwardOpenDataflowGraphView
    : virtual public IDownwardOpenDataflowGraphView {
  virtual std::unordered_set<DataflowGraphOutput> query_graph_outputs() const;
};

} // namespace FlexFlow

#endif

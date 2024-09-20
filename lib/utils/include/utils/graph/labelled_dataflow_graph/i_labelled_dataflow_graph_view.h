#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_I_LABELLED_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_I_LABELLED_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/dataflow_graph/i_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct ILabelledDataflowGraphView : virtual public IDataflowGraphView {
public:
  virtual NodeLabel at(Node const &) const = 0;
  virtual OutputLabel at(DataflowOutput const &) const = 0;

  virtual ~ILabelledDataflowGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledDataflowGraphView<int, int>);

} // namespace FlexFlow

#endif

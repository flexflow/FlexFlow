#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_I_LABELLED_OPEN_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_I_LABELLED_OPEN_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/labelled_dataflow_graph/i_labelled_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/i_open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
struct ILabelledOpenDataflowGraphView
    : virtual public ILabelledDataflowGraphView<NodeLabel, ValueLabel>,
      virtual public IOpenDataflowGraphView {
public:
  virtual NodeLabel const &at(Node const &) const override = 0;
  virtual ValueLabel const &at(OpenDataflowValue const &) const = 0;

  ValueLabel const &at(DataflowOutput const &o) const override final {
    return this->at(OpenDataflowValue{o});
  }

  virtual ~ILabelledOpenDataflowGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledOpenDataflowGraphView<int, int>);

} // namespace FlexFlow

#endif

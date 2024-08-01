#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_I_LABELLED_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_I_LABELLED_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/node_added_result.dtg.h"
#include "utils/graph/labelled_dataflow_graph/i_labelled_dataflow_graph_view.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct ILabelledDataflowGraph
    : virtual public ILabelledDataflowGraphView<NodeLabel, OutputLabel> {
public:
  virtual NodeAddedResult
      add_node(NodeLabel const &node_label,
               std::vector<DataflowOutput> const &inputs,
               std::vector<OutputLabel> const &output_labels) = 0;

  virtual void inplace_materialize_from(
      LabelledDataflowGraphView<NodeLabel, OutputLabel> const &) = 0;

  virtual ~ILabelledDataflowGraph() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledDataflowGraph<int, int>);

} // namespace FlexFlow

#endif

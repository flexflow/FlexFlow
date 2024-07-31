#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_I_LABELLED_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_I_LABELLED_OPEN_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/node_added_result.dtg.h"
#include "utils/graph/labelled_dataflow_graph/i_labelled_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/i_labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
struct ILabelledOpenDataflowGraph
    : virtual public ILabelledOpenDataflowGraphView<NodeLabel, ValueLabel>,
      virtual public ILabelledDataflowGraphView<NodeLabel, ValueLabel> {
  virtual NodeAddedResult
      add_node(NodeLabel const &node_label,
               std::vector<OpenDataflowValue> const &inputs,
               std::vector<ValueLabel> const &output_labels) = 0;

  virtual DataflowGraphInput add_input(ValueLabel const &value_label) = 0;

  virtual void inplace_materialize_from(LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &) = 0;

  // NodeAddedResult add_node(NodeLabel const &node_label,
  //                          std::vector<DataflowOutput> const &inputs,
  //                          std::vector<ValueLabel> const &output_labels)
  //                          override final {
  //   return this->add_node(node_label, transform(inputs, [](DataflowOutput
  //   const &o) { return OpenDataflowValue{o}; }), output_labels);
  // }

  virtual ~ILabelledOpenDataflowGraph() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledOpenDataflowGraph<int, int>);

} // namespace FlexFlow

#endif

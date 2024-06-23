#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_LABELLED_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_LABELLED_DATAFLOW_GRAPH_H

#include "utils/graph/labelled_dataflow_graph/i_labelled_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct LabelledDataflowGraph : virtual LabelledDataflowGraphView<NodeLabel, OutputLabel> {
private:
  using Interface = ILabelledDataflowGraph<NodeLabel, OutputLabel>;
public:
  NodeAddedResult add_node(NodeLabel const &node_label,
                           std::vector<DataflowOutput> const &inputs,
                           std::vector<OutputLabel> const &output_labels) {
    return this->get_interface().add_node(node_label, inputs, output_labels);
  }

private:
  Interface &get_interface() {
    return *std::dynamic_pointer_cast<Interface>(GraphView::ptr.get_mutable());
  }
  Interface const &get_interface() const {
    return *std::dynamic_pointer_cast<Interface const>(GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif

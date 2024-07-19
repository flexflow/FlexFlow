#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_LABELLED_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_LABELLED_DATAFLOW_GRAPH_H

#include "utils/graph/labelled_dataflow_graph/i_labelled_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct LabelledDataflowGraph
    : virtual LabelledDataflowGraphView<NodeLabel, OutputLabel> {
private:
  using Interface = ILabelledDataflowGraph<NodeLabel, OutputLabel>;

public:
  LabelledDataflowGraph(LabelledDataflowGraph const &) = default;
  LabelledDataflowGraph &operator=(LabelledDataflowGraph const &) = default;

  NodeAddedResult add_node(NodeLabel const &node_label,
                           std::vector<DataflowOutput> const &inputs,
                           std::vector<OutputLabel> const &output_labels) {
    return this->get_interface().add_node(node_label, inputs, output_labels);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, T>::value,
                                 LabelledDataflowGraph>::type
      create(Args &&...args) {
    return LabelledDataflowGraph(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

  template <typename T>
  static typename std::enable_if<std::is_base_of<Interface, T>::value,
                                 LabelledDataflowGraph>::type
      create_copy_of(
          LabelledDataflowGraphView<NodeLabel, OutputLabel> const &view) {
    cow_ptr_t<T> impl = make_cow_ptr<T>();
    impl.get_mutable()->inplace_materialize_from(view);
    return LabelledDataflowGraph(std::move(impl));
  }

protected:
  using LabelledDataflowGraphView<NodeLabel,
                                  OutputLabel>::LabelledDataflowGraphView;

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

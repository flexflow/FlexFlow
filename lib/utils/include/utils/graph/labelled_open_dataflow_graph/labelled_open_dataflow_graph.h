#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_H

#include "utils/graph/labelled_open_dataflow_graph/i_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
struct LabelledOpenDataflowGraph
    : virtual public LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> {
private:
  using Interface = ILabelledOpenDataflowGraph<NodeLabel, ValueLabel>;

public:
  LabelledOpenDataflowGraph(LabelledOpenDataflowGraph const &) = default;
  LabelledOpenDataflowGraph &
      operator=(LabelledOpenDataflowGraph const &) = default;

  NodeAddedResult add_node(NodeLabel const &node_label,
                           std::vector<OpenDataflowValue> const &inputs,
                           std::vector<ValueLabel> const &output_labels) {
    return this->get_interface().add_node(node_label, inputs, output_labels);
  }

  DataflowGraphInput add_input(ValueLabel const &value_label) {
    return this->get_interface().add_input(value_label);
  }

  template <typename T>
  static typename std::enable_if<std::is_base_of<Interface, T>::value,
                                 LabelledOpenDataflowGraph>::type
      create() {
    return LabelledOpenDataflowGraph(make_cow_ptr<T>());
  }

  template <typename T>
  static std::enable_if_t<std::is_base_of_v<Interface, T>, LabelledOpenDataflowGraph>
    create_copy_of(LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &view) {
    cow_ptr_t<T> impl = make_cow_ptr<T>();
    impl.get_mutable()->inplace_materialize_from(view);
    return LabelledOpenDataflowGraph(std::move(impl));
  }

protected:
  using LabelledOpenDataflowGraphView<NodeLabel, ValueLabel>::
      LabelledOpenDataflowGraphView;

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

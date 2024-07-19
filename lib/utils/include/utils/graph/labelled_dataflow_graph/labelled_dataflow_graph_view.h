#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_LABELLED_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_LABELLED_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/labelled_dataflow_graph/i_labelled_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct LabelledDataflowGraphView : virtual public DataflowGraphView {
private:
  using Interface = ILabelledDataflowGraphView<NodeLabel, OutputLabel>;

public:
  LabelledDataflowGraphView(LabelledDataflowGraphView const &) = default;
  LabelledDataflowGraphView &
      operator=(LabelledDataflowGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return this->get_interface().at(n);
  }
  OutputLabel const &at(DataflowOutput const &o) const {
    return this->get_interface().at(o);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, T>::value,
                                 LabelledDataflowGraphView>::type
      create(Args &&...args) {
    return LabelledDataflowGraphView(
        make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using DataflowGraphView::DataflowGraphView;

private:
  Interface const &get_interface() const {
    return *std::dynamic_pointer_cast<Interface const>(GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif

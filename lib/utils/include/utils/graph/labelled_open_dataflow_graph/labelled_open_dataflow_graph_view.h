#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"
#include "utils/graph/labelled_open_dataflow_graph/i_labelled_open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
struct LabelledOpenDataflowGraphView
    : virtual public LabelledDataflowGraphView<NodeLabel, ValueLabel>,
      virtual public OpenDataflowGraphView {
private:
  using Interface = ILabelledOpenDataflowGraphView<NodeLabel, ValueLabel>;

public:
  LabelledOpenDataflowGraphView(LabelledOpenDataflowGraphView const &) =
      default;
  LabelledOpenDataflowGraphView &
      operator=(LabelledOpenDataflowGraphView const &) = default;

  NodeLabel const &at(Node const &n) const {
    return this->get_interface().at(n);
  }

  ValueLabel const &at(OpenDataflowValue const &v) const {
    return this->get_interface().at(v);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<Interface, T>::value,
      LabelledOpenDataflowGraphView<NodeLabel, ValueLabel>>::type
      create(Args &&...args) {
    return LabelledOpenDataflowGraphView(static_cast<cow_ptr_t<IGraphView>>(
        make_cow_ptr<T>(std::forward<Args>(args)...)));
  }

protected:
  using OpenDataflowGraphView::OpenDataflowGraphView;
  // using LabelledDataflowGraphView<NodeLabel,
  // ValueLabel>::LabelledDataflowGraphView;
private:
  Interface const &get_interface() const {
    return *std::dynamic_pointer_cast<Interface const>(GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif

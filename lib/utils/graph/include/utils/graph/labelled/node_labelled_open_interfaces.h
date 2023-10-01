#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_OPEN_INTERFACES
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_OPEN_INTERFACES

#include "node_labelled_interfaces.h"
#include "utils/graph/open_graph_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel>
struct INodeLabelledOpenMultiDiGraphView : virtual INodeLabelledMultiDiGraphView<NodeLabel>,
                                            virtual OpenMultiDiGraphView {
  INodeLabelledOpenMultiDiGraphView(INodeLabelledOpenMultiDiGraphView const &) = delete;
  INodeLabelledOpenMultiDiGraphView &
      operator=(INodeLabelledOpenMultiDiGraphView const &) = delete;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(INodeLabelledOpenMultiDiGraphView<int>);

template <typename NodeLabel>
struct INodeLabelledOpenMultiDiGraph : virtual INodeLabelledOpenMultiDiGraphView<NodeLabel> {
  Node add_node(Node const &node) = 0;
  void add_edge(OpenMultiDiEdge cosnt &e) = 0;
}

}

#endif

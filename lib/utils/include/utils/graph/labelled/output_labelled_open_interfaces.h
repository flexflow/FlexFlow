#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN_INTERFACES
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN_INTERFACES

#include "node_labelled_open.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
struct IOutputLabelledOpenMultiDiGraphView
    : virtual INodeLabelledOpenMultiDiGraphView<NodeLabel> {
  virtual EdgeLabel const &at(InputMultiDiEdge const &) const = 0;
  virtual EdgeLabel const &at(MultiDiOutput const &) const = 0;

  using INodeLabelledOpenMultiDiGraphView<NodeLabel>::at;
};

template <typename NodeLabel, typename EdgeLabel>
struct IOutputLabelledOpenMultiDiGraph
    : virtual public IOutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> {
  virtual EdgeLabel &at(InputMultiDiEdge const &) = 0;
  virtual EdgeLabel &at(MultiDiOutput const &) = 0;
  virtual Node add_node(NodeLabel const &) = 0;
  virtual NodePort add_node_port() = 0;
  virtual NodeLabel &at(Node const &) = 0;
  virtual void add_label(MultiDiOutput const &o, EdgeLabel const &l) = 0;
  virtual void add_label(InputMultiDiEdge const &e, EdgeLabel const &l) = 0;
  virtual void add_edge(OpenMultiDiEdge const &e) = 0;

  using IOutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel>::at;
};

} // namespace FlexFlow

#endif

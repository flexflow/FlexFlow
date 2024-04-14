#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_GRAPH_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_GRAPH_INTERFACES_H

#include "node_labelled_open.h"
#include "utils/graph/open_graphs.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct IOutputLabelledMultiDiGraphView
    : public INodeLabelledMultiDiGraphView<NodeLabel> {

  virtual OutputLabel const &at(MultiDiOutput const &) const = 0;

  using INodeLabelledMultiDiGraphView<NodeLabel>::at;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOutputLabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename OutputLabel>
struct IOutputLabelledMultiDiGraph
    : public IOutputLabelledMultiDiGraphView<NodeLabel, OutputLabel>,
      public INodeLabelledMultiDiGraph<NodeLabel> {
public:
  virtual IOutputLabelledMultiDiGraph *clone() const = 0;

  virtual void add_output(MultiDiOutput const &output,
                          OutputLabel const &label) = 0;
  virtual NodePort add_node_port() = 0;

  virtual NodeLabel &at(Node const &) = 0;
  virtual NodeLabel const &at(Node const &) const = 0;
  virtual OutputLabel &at(MultiDiOutput const &) = 0;
  virtual OutputLabel const &at(MultiDiOutput const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOutputLabelledMultiDiGraph<int, int>);

} // namespace FlexFlow

#endif

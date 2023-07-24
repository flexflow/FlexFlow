#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_GRAPH_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_GRAPH_INTERFACES_H

#include "node_labelled_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel>
struct IOutputLabelledMultiDiGraphView
    : public INodeLabelledMultiDiGraphView<NodeLabel> {

  virtual OutputLabel &at(MultiDiOutput const &) = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOutputLabelledMultiDiGraphView<int, int>);

template <typename NodeLabel, typename OutputLabel>
struct IOutputLabelledMultiDiGraph
    : public IOutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> {
public:
  virtual IOutputLabelledMultiDiGraph *clone() const = 0;

  virtual void add_output(MultiDiOutput const &output,
                          OutputLabel const &label) = 0;
  virtual void add_edge(MultiDiOutput const &output,
                        MultiDiInput const &input) = 0;

  virtual NodeLabel &at(Node const &) = 0;
  virtual NodeLabel const &at(Node const &) const = 0;
  virtual OutputLabel const &at(MultiDiOutput const &) const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOutputLabelledMultiDiGraph<int, int>);

} // namespace FlexFlow

#endif

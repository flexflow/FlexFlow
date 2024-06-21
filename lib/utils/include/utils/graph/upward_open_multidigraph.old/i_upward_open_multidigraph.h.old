#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_I_UPWARD_OPEN_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_I_UPWARD_OPEN_MULTIDIGRAPH_H

#include "utils/graph/upward_open_multidigraph/i_upward_open_multidigraph_view.h"

namespace FlexFlow {

struct IUpwardOpenMultiDiGraph : virtual public IUpwardOpenMultiDiGraphView {
  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &node) = 0;
  virtual void remove_node_unsafe(Node const &node) = 0;
  virtual void add_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual IUpwardOpenMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUpwardOpenMultiDiGraph);

} // namespace FlexFlow

#endif

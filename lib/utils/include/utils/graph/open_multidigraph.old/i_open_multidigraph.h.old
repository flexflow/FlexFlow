#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_I_OPEN_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_I_OPEN_MULTIDIGRAPH_H

#include "utils/graph/open_multidigraph/i_open_multidigraph_view.h"

namespace FlexFlow {

struct IOpenMultiDiGraph : virtual public IOpenMultiDiGraphView {
  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &node) = 0;
  virtual void remove_node_unsafe(Node const &node) = 0;
  virtual void add_edge(OpenMultiDiEdge const &) = 0;
  virtual void remove_edge(OpenMultiDiEdge const &) = 0;
  virtual IOpenMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenMultiDiGraph);

} // namespace FlexFlow

#endif

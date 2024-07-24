#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_I_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_I_MULTIDIGRAPH_H

#include "utils/graph/multidigraph/i_multidigraph_view.h"

namespace FlexFlow {

struct IMultiDiGraph : virtual public IMultiDiGraphView {
  virtual Node add_node() = 0;
  virtual void remove_node(Node const &) = 0;
  virtual MultiDiEdge add_edge(Node const &src, Node const &dst) = 0;
  virtual void remove_edge(MultiDiEdge const &) = 0;
  virtual IMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IMultiDiGraph);

} // namespace FlexFlow

#endif

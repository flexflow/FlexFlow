#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_I_DOWNWARD_OPEN_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_I_DOWNWARD_OPEN_MULTIDIGRAPH_H

#include "utils/graph/downward_open_multidigraph/i_downward_open_multidigraph_view.h"

namespace FlexFlow {

struct IDownwardOpenMultiDiGraph
    : virtual public IDownwardOpenMultiDiGraphView {
  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &node) = 0;
  virtual void remove_node_unsafe(Node const &node) = 0;
  virtual void add_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual IDownwardOpenMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDownwardOpenMultiDiGraph);


} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_I_DIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_I_DIGRAPH_H

#include "utils/graph/digraph/i_digraph_view.h"

namespace FlexFlow {

struct IDiGraph : virtual public IDiGraphView {
  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &) = 0;
  virtual void remove_node_unsafe(Node const &) = 0;
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;
  virtual IDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDiGraph);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_I_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_I_GRAPH_H

#include "utils/graph/node/i_graph_view.h"
#include "utils/graph/node/node.dtg.h"

namespace FlexFlow {

struct IGraph : virtual IGraphView {
  IGraph() = default;
  IGraph(IGraph const &) = delete;
  IGraph &operator=(IGraph const &) = delete;

  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &) = 0;
  virtual void remove_node_unsafe(Node const &) = 0;
  virtual IGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IGraph);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_I_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_I_MULTIDIGRAPH_H

#include "utils/graph/multidigraph/i_multidigraph_view.h"

namespace FlexFlow {

struct IMultiDiGraph : virtual public IMultiDiGraphView {
  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &) = 0;
  virtual void remove_node_unsafe(Node const &) = 0;
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;

  virtual std::unordered_set<Node>
      query_nodes(NodeQuery const &query) const override {
    return static_cast<IMultiDiGraphView const *>(this)->query_nodes(query);
  }

  virtual IMultiDiGraph *clone() const override = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IMultiDiGraph);

} // namespace FlexFlow

#endif

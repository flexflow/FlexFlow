#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_I_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_I_GRAPH_VIEW_H

#include "utils/graph/node/node_query.dtg.h"
#include "utils/type_traits.h"

namespace FlexFlow {

struct IGraphView {
  IGraphView() = default;
  IGraphView(IGraphView const &) = delete;
  IGraphView &operator=(IGraphView const &) = delete;

  virtual IGraphView *clone() const = 0;

  virtual std::unordered_set<Node> query_nodes(NodeQuery const &) const = 0;
  virtual ~IGraphView(){};
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IGraphView);

} // namespace FlexFlow

#endif

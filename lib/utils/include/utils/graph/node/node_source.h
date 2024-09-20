#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_NODE_NODE_SOURCE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_NODE_NODE_SOURCE_H

#include "utils/graph/node/node.dtg.h"

namespace FlexFlow {

struct NodeSource {
public:
  NodeSource();

  Node new_node();

private:
  static size_t next_available_node_id;
};

} // namespace FlexFlow

#endif

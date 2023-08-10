#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_DIRECTED_EDGE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_DIRECTED_EDGE_H

#include "node.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct DirectedEdge {
  Node src;
  Node dst;
};
FF_VISITABLE_STRUCT(DirectedEdge, src, dst);

}

#endif

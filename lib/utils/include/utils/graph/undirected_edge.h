#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_EDGE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_EDGE_H

#include "node.h"

namespace FlexFlow {

struct UndirectedEdge {
  Node smaller;
  Node bigger;
};
FF_VISITABLE_STRUCT(UndirectedEdge, smaller, bigger);

}

#endif

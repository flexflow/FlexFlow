#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_NODE_ALGORITHMS_GENERATE_NEW_NODE_ID_PERMUTATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_NODE_ALGORITHMS_GENERATE_NEW_NODE_ID_PERMUTATION_H

#include "utils/graph/node/algorithms/new_node.dtg.h"
#include "utils/graph/node/graph_view.h"

namespace FlexFlow {

bidict<NewNode, Node> generate_new_node_id_permutation(GraphView const &);

} // namespace FlexFlow

#endif

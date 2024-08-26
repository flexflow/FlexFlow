#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_NODE_IDS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_NODE_IDS_H

#include "utils/graph/node/algorithms/new_node.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

OpenDataflowGraphView
    permute_node_ids(OpenDataflowGraphView const &,
                     bidict<NewNode, Node> const &new_node_tofrom_old_node);

} // namespace FlexFlow

#endif

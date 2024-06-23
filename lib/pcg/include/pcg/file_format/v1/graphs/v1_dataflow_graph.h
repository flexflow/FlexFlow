#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_MULTIDIGRAPH_H

#include "pcg/file_format/v1/graphs/v1_dataflow_graph.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_graph_view.h"

namespace FlexFlow {

V1DataflowGraph to_v1(DataflowGraphView const &);
V1DataflowGraph to_v1(DataflowGraphView const &,
                      std::unordered_map<Node, size_t> const &);

} // namespace FlexFlow

#endif

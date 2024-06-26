#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_MULTIDIGRAPH_H

#include "pcg/file_format/v1/graphs/v1_multidigraph.dtg.h"
#include "utils/graph.h"

namespace FlexFlow {

V1MultiDiGraph to_v1(MultiDiGraphView const &);
V1MultiDiGraph to_v1(MultiDiGraphView const &,
                     std::unordered_map<Node, size_t> const &,
                     std::unordered_map<NodePort, size_t> const &);

} // namespace FlexFlow

#endif

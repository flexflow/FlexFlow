#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_COMPUTATION_GRAPH_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/file_format/v1/v1_computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"

namespace FlexFlow {

V1ComputationGraph to_v1(ComputationGraph const &);

std::pair<V1ComputationGraph, bidict<int, layer_guid_t>>
    to_v1_including_node_numbering(ComputationGraph const &);

} // namespace FlexFlow

#endif

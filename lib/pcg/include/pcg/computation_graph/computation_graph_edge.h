#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_COMPUTATION_GRAPH_EDGE_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_COMPUTATION_GRAPH_EDGE_H

#include "pcg/computation_graph/computation_graph_edge.dtg.h"
#include "pcg/layer_guid_t.dtg.h"

namespace FlexFlow {

layer_guid_t get_computation_graph_edge_src_layer(ComputationGraphEdge const &);
layer_guid_t get_computation_graph_edge_dst_layer(ComputationGraphEdge const &);

} // namespace FlexFlow

#endif

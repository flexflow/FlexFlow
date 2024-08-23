#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_COMPUTATION_GRAPH_EDGE_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_COMPUTATION_GRAPH_EDGE_H

#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

parallel_tensor_guid_t get_parallel_tensor(ParallelComputationGraphEdge const &);
parallel_layer_guid_t get_src_layer(ParallelComputationGraphEdge const &);
parallel_layer_guid_t get_dst_layer(ParallelComputationGraphEdge const &);
int get_dst_layer_input_idx(ParallelComputationGraphEdge const &);

} // namespace FlexFlow

#endif

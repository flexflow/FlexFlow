#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"

namespace FlexFlow {

parallel_tensor_guid_t get_parallel_tensor(ParallelComputationGraphEdge const &e) {
  return parallel_tensor_guid_t{e.raw_edge.src}; 
}

parallel_layer_guid_t get_src_layer(ParallelComputationGraphEdge const &e) {
  return parallel_layer_guid_t{e.raw_edge.src.node};
}

parallel_layer_guid_t get_dst_layer(ParallelComputationGraphEdge const &e) {
  return parallel_layer_guid_t{e.raw_edge.dst.node};
}

int get_dst_layer_input_idx(ParallelComputationGraphEdge const &e) {
  return e.raw_edge.dst.idx;
}

} // namespace FlexFlow

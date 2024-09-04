#include "pcg/computation_graph/computation_graph_edge.h"

namespace FlexFlow {

layer_guid_t get_computation_graph_edge_src_layer(ComputationGraphEdge const &e) {
  return layer_guid_t{e.raw_edge.src.node};
}

layer_guid_t get_computation_graph_edge_dst_layer(ComputationGraphEdge const &e) {
  return layer_guid_t{e.raw_edge.dst.node};
}

} // namespace FlexFlow

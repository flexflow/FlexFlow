#include "pcg/file_format/v1/v1_computation_graph.h"
#include "pcg/file_format/v1/graphs/v1_labelled_dataflow_graph.h"

namespace FlexFlow {

V1ComputationGraph to_v1(ComputationGraph const &g) {
  return V1ComputationGraph{
    to_v1<LayerAttrs, TensorAttrs>(g.raw_graph),
  };
}

std::pair<
  V1ComputationGraph,
  bidict<int, layer_guid_t>
> to_v1_including_node_numbering(ComputationGraph const &cg) {
  std::pair<
    V1LabelledDataflowGraph<LayerAttrs, TensorAttrs>,
    bidict<int, Node>
  > raw = to_v1_including_node_numbering<LayerAttrs, TensorAttrs>(cg.raw_graph);
  V1ComputationGraph v1_cg = V1ComputationGraph{raw.first};
  bidict<int, layer_guid_t> v1_node_ids = map_values(raw.second, [](Node const &n) { return layer_guid_t{n}; });

  return {v1_cg, v1_node_ids};
}

} // namespace FlexFlow

#include "pcg/parallel_computation_graph.h"
#include "utils/containers.h"

namespace FlexFlow {

ParallelComputationGraph empty_parallel_computation_graph() {
  return ParallelComputationGraph{DataflowGraph<ParallelLayerAttrs, ParallelTensorAttrs>{}};
}

std::unordered_set<parallel_layer_guid_t> get_parallel_layers(ParallelComputationGraph const &pcg) {
  return transform(get_nodes(pcg.raw_graph),
                   [&](Node const &n) { return parallel_layer_guid_t{n}; });
}

ParallelTensorAttrs get_parallel_tensor_attrs(ParallelComputationGraph const &pcg,
                                              parallel_tensor_guid_t const &t) {
  return pcg.raw_graph.at(t.raw_graph_output);
}

} // namespace FlexFlow

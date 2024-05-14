#include "pcg/computation_graph.h"
#include "utils/containers.h"

namespace FlexFlow {

ComputationGraph make_empty_computation_graph() {
  return ComputationGraph{
    DataflowGraph<LayerAttrs, TensorAttrs>{}
  };
}

std::unordered_set<layer_guid_t> get_layers(ComputationGraph const &cg) {
  return transform(get_nodes(cg.raw_graph), [&](Node const &n) { return layer_guid_t{n}; });
}

TensorAttrs get_tensor_attrs(ComputationGraph const &cg, tensor_guid_t const &t) {
  return cg.raw_graph.at(t.raw_graph_output);
}

} // namespace FlexFlow

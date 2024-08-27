#include "pcg/computation_graph.h"
#include "utils/containers/get_only.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

ComputationGraph make_empty_computation_graph() {
  return ComputationGraph{
      LabelledDataflowGraph<LayerAttrs, TensorAttrs>::create<
          UnorderedSetLabelledOpenDataflowGraph<LayerAttrs, TensorAttrs>>()};
}

std::unordered_set<layer_guid_t> get_layers(ComputationGraph const &cg) {
  return transform(get_nodes(cg.raw_graph),
                   [&](Node const &n) { return layer_guid_t{n}; });
}

TensorAttrs get_tensor_attrs(ComputationGraph const &cg,
                             tensor_guid_t const &t) {
  return cg.raw_graph.at(t.raw_graph_output);
}

std::vector<layer_guid_t> topological_ordering(ComputationGraph const &cg) {
  std::vector<Node> layers = get_topological_ordering(cg.raw_graph);
  return transform(
      layers, [&](Node const &e) -> layer_guid_t { return layer_guid_t{e}; });
}

std::vector<layer_guid_t>
    reverse_topological_ordering(ComputationGraph const &cg) {
  std::vector<Node> layers =
      reversed<std::vector<Node>>(get_topological_ordering(cg.raw_graph));
  return transform(
      layers, [&](Node const &e) -> layer_guid_t { return layer_guid_t{e}; });
}

std::vector<tensor_guid_t> get_outgoing_tensors(ComputationGraph const &cg,
                                                layer_guid_t n) {
  return transform(get_outputs(cg.raw_graph, n.raw_node),
                   [](DataflowOutput const &o) { return tensor_guid_t{o}; });
}

std::vector<tensor_guid_t> get_incoming_tensors(ComputationGraph const &cg,
                                                layer_guid_t n) {
  return transform(get_inputs(cg.raw_graph, n.raw_node),
                   [](DataflowOutput const &o) { return tensor_guid_t{o}; });
}

LayerAttrs get_layer_attrs(ComputationGraph const &cg, layer_guid_t const &n) {
  return cg.raw_graph.at(n.raw_node);
}

layer_guid_t get_layer_by_name(ComputationGraph const &cg,
                               std::string const &name) {
  std::unordered_set<layer_guid_t> found =
      filter(get_layers(cg), [&](layer_guid_t const &l) {
        return get_layer_attrs(cg, l).name == name;
      });
  return get_only(found);
}

std::vector<tensor_guid_t>
    get_new_tensor_guids_for_layer_without_graph_insertion(
        ComputationGraph const &cg, layer_guid_t const &n, int num_tensors) {
  std::vector<tensor_guid_t> new_tensor_guids;
  int num_outgoing_tensors = get_outgoing_tensors(cg, n).size();

  for (int i = 0; i < num_tensors; ++i) {
    new_tensor_guids.push_back(
        tensor_guid_t{DataflowOutput{n.raw_node, num_outgoing_tensors + i}});
  }
  return new_tensor_guids;
}

} // namespace FlexFlow

#include "pcg/computation_graph.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "op-attrs/get_incoming_tensor_roles.h"

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

LayerAddedResult add_layer(ComputationGraph &computation_graph,
                           LayerAttrs const &attrs,
                           std::vector<tensor_guid_t> const &inputs,
                           std::vector<TensorAttrs> const &outputs) {
  NodeAddedResult raw_added = computation_graph.raw_graph.add_node(attrs,
                                                                   transform(inputs, [](tensor_guid_t const &t) { return t.raw_graph_output; }),
                                                                   outputs);
  return LayerAddedResult{
    layer_guid_t{raw_added.node},
    transform(raw_added.outputs, [](DataflowOutput const &o) { return tensor_guid_t{o}; }),
  };
}

TensorAttrs get_tensor_attrs(ComputationGraph const &cg,
                             tensor_guid_t const &t) {
  return cg.raw_graph.at(t.raw_graph_output);
}

bool are_tensor_guid_shapes_equivalent(ComputationGraph const &cg,
                                       tensor_guid_t const &t1,
                                       tensor_guid_t const &t2) {
  return get_tensor_attrs(cg, t1).shape == get_tensor_attrs(cg, t2).shape;
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
  return transform(get_input_values(cg.raw_graph, n.raw_node),
                   [](DataflowOutput const &o) { return tensor_guid_t{o}; });
}

static std::vector<tensor_guid_t> get_incoming_tensors_with_role(ComputationGraph const &cg, layer_guid_t const &l, IncomingTensorRole desired_role) {
  ComputationGraphOpAttrs attrs = get_layer_attrs(cg, l).attrs;

  std::vector<tensor_guid_t> incoming_tensors = get_incoming_tensors(cg, l);

  std::vector<IncomingTensorRole> incoming_tensor_roles = get_incoming_tensor_roles(attrs, incoming_tensors.size());

  assert (incoming_tensors.size() == incoming_tensor_roles.size());

  std::vector<tensor_guid_t> result = filtrans(zip(incoming_tensors, incoming_tensor_roles), 
                                               [&](std::pair<tensor_guid_t, IncomingTensorRole> const &p) -> std::optional<tensor_guid_t> {
                                                 tensor_guid_t tensor = p.first;
                                                 IncomingTensorRole role = p.second;

                                                 if (role == desired_role) {
                                                   return tensor;
                                                 } else {
                                                   return std::nullopt;
                                                 }
                                               });
  return result;
}

std::vector<tensor_guid_t> get_incoming_inputs(ComputationGraph const &cg, layer_guid_t const &l) {
  return get_incoming_tensors_with_role(cg, l, IncomingTensorRole::INPUT);
}

std::vector<tensor_guid_t> get_incoming_weights(ComputationGraph const &cg, layer_guid_t const &l) {
  return get_incoming_tensors_with_role(cg, l, IncomingTensorRole::WEIGHT);
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

} // namespace FlexFlow

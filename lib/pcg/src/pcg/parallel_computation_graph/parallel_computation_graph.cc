#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "op-attrs/get_incoming_tensor_roles.h"

namespace FlexFlow {

ParallelComputationGraph empty_parallel_computation_graph() {
  return ParallelComputationGraph{
      LabelledDataflowGraph<ParallelLayerAttrs, ParallelTensorAttrs>::create<
          UnorderedSetLabelledOpenDataflowGraph<ParallelLayerAttrs,
                                                ParallelTensorAttrs>>()};
}

std::unordered_set<parallel_layer_guid_t>
    get_parallel_layers(ParallelComputationGraph const &pcg) {
  return transform(get_nodes(pcg.raw_graph),
                   [&](Node const &n) { return parallel_layer_guid_t{n}; });
}

ParallelLayerAddedResult
    add_parallel_layer(ParallelComputationGraph &pcg,
                       ParallelLayerAttrs const &layer_attrs,
                       std::vector<parallel_tensor_guid_t> const &inputs,
                       std::vector<ParallelTensorAttrs> const &output_labels) {
  std::vector<DataflowOutput> unwrapped_inputs =
      transform(inputs, [](parallel_tensor_guid_t const &t) {
        return t.raw_graph_output;
      });
  NodeAddedResult op_added =
      pcg.raw_graph.add_node(layer_attrs, unwrapped_inputs, output_labels);
  return ParallelLayerAddedResult{
      parallel_layer_guid_t{op_added.node},
      transform(
          op_added.outputs,
          [](DataflowOutput const &o) { return parallel_tensor_guid_t{o}; }),
  };
}

std::vector<parallel_tensor_guid_t>
    get_incoming_tensors(ParallelComputationGraph const &pcg,
                         parallel_layer_guid_t const &l) {
  return transform(
      get_input_values(pcg.raw_graph, l.raw_graph_node),
      [](DataflowOutput const &o) { return parallel_tensor_guid_t{o}; });
}

std::vector<parallel_tensor_guid_t>
    get_layer_outputs(ParallelComputationGraph const &pcg,
                      parallel_layer_guid_t const &l) {
  return transform(
      get_outputs(pcg.raw_graph, l.raw_graph_node),
      [](DataflowOutput const &o) { return parallel_tensor_guid_t{o}; });
}

static std::vector<parallel_tensor_guid_t> get_incoming_tensors_with_role(ParallelComputationGraph const &pcg, 
                                                                          parallel_layer_guid_t const &l,
                                                                          IncomingTensorRole desired_role) {
  PCGOperatorAttrs attrs = get_parallel_layer_attrs(pcg, l).op_attrs;

  std::vector<parallel_tensor_guid_t> incoming_tensors = get_incoming_tensors(pcg, l);

  std::vector<IncomingTensorRole> incoming_tensor_roles = get_incoming_tensor_roles(attrs, incoming_tensors.size());

  assert (incoming_tensors.size() == incoming_tensor_roles.size());

  std::vector<parallel_tensor_guid_t> result = filtrans(zip(incoming_tensors, incoming_tensor_roles), 
                                                        [&](std::pair<parallel_tensor_guid_t, IncomingTensorRole> const &p) -> std::optional<parallel_tensor_guid_t> {
                                                          parallel_tensor_guid_t tensor = p.first;
                                                          IncomingTensorRole role = p.second;

                                                          if (role == desired_role) {
                                                            return tensor;
                                                          } else {
                                                            return std::nullopt;
                                                          }
                                                        });
  return result;
}

std::vector<parallel_tensor_guid_t> get_incoming_inputs(ParallelComputationGraph const &pcg, 
                                                        parallel_layer_guid_t const &l) {
  return get_incoming_tensors_with_role(pcg, l, IncomingTensorRole::INPUT);
}

std::vector<parallel_tensor_guid_t> get_incoming_weights(ParallelComputationGraph const &pcg, 
                                                         parallel_layer_guid_t const &l) {
  return get_incoming_tensors_with_role(pcg, l, IncomingTensorRole::WEIGHT);
}

parallel_layer_guid_t get_source_layer(ParallelComputationGraph const &g,
                                       parallel_tensor_guid_t const &t) {
  return parallel_layer_guid_t{t.raw_graph_output.node};
}

ParallelLayerAttrs get_parallel_layer_attrs(ParallelComputationGraph const &pcg,
                                            parallel_layer_guid_t const &l) {
  return pcg.raw_graph.at(l.raw_graph_node);
}

ParallelTensorAttrs
    get_parallel_tensor_attrs(ParallelComputationGraph const &pcg,
                              parallel_tensor_guid_t const &t) {
  return pcg.raw_graph.at(t.raw_graph_output);
}

std::vector<parallel_layer_guid_t>
    topological_ordering(ParallelComputationGraph const &pcg) {
  return transform(get_topological_ordering(pcg.raw_graph),
                   [](Node const &n) { return parallel_layer_guid_t{n}; });
}

parallel_layer_guid_t
    get_parallel_layer_by_name(ParallelComputationGraph const &pcg,
                               std::string const &name) {
  std::unordered_set<parallel_layer_guid_t> found =
      filter(get_parallel_layers(pcg), [&](parallel_layer_guid_t const &l) {
        return get_parallel_layer_attrs(pcg, l).name == name;
      });
  return get_only(found);
}

} // namespace FlexFlow

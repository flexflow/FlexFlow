#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/dataflow_graph/algorithms.h"
#include "utils/containers.h"

namespace FlexFlow {

ParallelComputationGraph empty_parallel_computation_graph() {
  return ParallelComputationGraph{
      DataflowGraph<ParallelLayerAttrs, ParallelTensorAttrs>{}};
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
  std::vector<MultiDiOutput> unwrapped_inputs =
      transform(inputs, [](parallel_tensor_guid_t const &t) {
        return t.raw_graph_output;
      });
  OperatorAddedResult op_added =
      pcg.raw_graph.add_operator(layer_attrs, unwrapped_inputs, output_labels);
  return ParallelLayerAddedResult{
      parallel_layer_guid_t{op_added.node},
      transform(
          op_added.outputs,
          [](MultiDiOutput const &o) { return parallel_tensor_guid_t{o}; }),
  };
}

std::vector<parallel_tensor_guid_t>
    get_layer_inputs(ParallelComputationGraph const &pcg,
                     parallel_layer_guid_t const &l) {
  return transform(
      get_inputs(pcg.raw_graph, l.raw_graph_node),
      [](MultiDiOutput const &o) { return parallel_tensor_guid_t{o}; });
}

std::vector<parallel_tensor_guid_t>
    get_layer_outputs(ParallelComputationGraph const &pcg,
                      parallel_layer_guid_t const &l) {
  return transform(
      get_outputs(pcg.raw_graph, l.raw_graph_node),
      [](MultiDiOutput const &o) { return parallel_tensor_guid_t{o}; });
}

parallel_layer_guid_t get_source_layer(ParallelComputationGraph const &g,
                                       parallel_tensor_guid_t const &t) {
  return parallel_layer_guid_t{t.raw_graph_output.src};
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
  return transform(topological_ordering(pcg.raw_graph),
                   [](Node const &n) { return parallel_layer_guid_t{n}; });
}

} // namespace FlexFlow

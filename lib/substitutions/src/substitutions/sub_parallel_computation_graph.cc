#include "substitutions/sub_parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/create_lazy_copy_of_labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"

namespace FlexFlow {

std::unordered_set<parallel_layer_guid_t>
    get_parallel_layers(SubParallelComputationGraph const &sub_pcg) {
  return transform(get_nodes(sub_pcg.raw_graph), [](Node const &n) { return parallel_layer_guid_t{n}; });
}

ParallelLayerAttrs
    get_parallel_layer_attrs(SubParallelComputationGraph const &spcg,
                             parallel_layer_guid_t const &layer) {
  return spcg.raw_graph.at(layer.raw_graph_node);
}

PCGOperatorAttrs get_operator_attrs(SubParallelComputationGraph const &spcg,
                                    parallel_layer_guid_t const &n) {
  return get_parallel_layer_attrs(spcg, n).op_attrs;
}

ParallelTensorAttrs
    get_parallel_tensor_attrs(SubParallelComputationGraph const &spcg,
                              open_parallel_tensor_guid_t const &v) {
  return spcg.raw_graph.at(v.raw_open_dataflow_value);
}

SubParallelComputationGraph
    sub_pcg_from_full_pcg(ParallelComputationGraph const &pcg) {
  return SubParallelComputationGraph{
      view_as_labelled_open_dataflow_graph(pcg.raw_graph)};
}

ParallelComputationGraph pcg_from_sub_pcg_by_dropping_inputs(
    SubParallelComputationGraph const &sub_pcg) {
  return ParallelComputationGraph{
      LabelledDataflowGraph<ParallelLayerAttrs, ParallelTensorAttrs>::
          create_copy_of<
              UnorderedSetLabelledOpenDataflowGraph<ParallelLayerAttrs,
                                                    ParallelTensorAttrs>>(
              sub_pcg.raw_graph)};
  // return ParallelComputationGraph{
  //   make_lazy_copy_of<
  //     UnorderedSetLabelledOpenDataflowGraph<ParallelLayerAttrs,
  //     ParallelTensorAttrs>
  //     >(sub_pcg.raw_graph)
  // };
}

parallel_layer_guid_t
    get_parallel_layer_by_name(SubParallelComputationGraph const &pcg,
                               std::string const &name) {
  return get_parallel_layer_by_name(pcg_from_sub_pcg_by_dropping_inputs(pcg),
                                    name);
}

std::vector<open_parallel_tensor_guid_t>
    get_layer_inputs(SubParallelComputationGraph const &pcg,
                     parallel_layer_guid_t const &layer) {
  return transform(get_inputs(pcg.raw_graph, layer.raw_graph_node),
                   [](OpenDataflowValue const &v) { return open_parallel_tensor_guid_t{v}; });
}

std::vector<parallel_tensor_guid_t>
    get_layer_outputs(SubParallelComputationGraph const &pcg,
                      parallel_layer_guid_t const &layer) {
  return transform(get_outputs(pcg.raw_graph, layer.raw_graph_node),
                   [](DataflowOutput const &o) { return parallel_tensor_guid_t{o}; });
}

SubParallelComputationGraphData get_graph_data(SubParallelComputationGraph const &pcg) {
  LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorAttrs> raw_data = get_graph_data(pcg.raw_graph);

  return SubParallelComputationGraphData{
    map_keys(raw_data.node_data, [](Node const &n) { return parallel_layer_guid_t{n}; }),
    raw_data.edges,
    transform(raw_data.inputs, [](DataflowGraphInput const &i) { return input_parallel_tensor_guid_t{i}; }),
    map_keys(raw_data.value_data, [](OpenDataflowValue const &v) { return open_parallel_tensor_guid_t{v}; }),
  };
}

bool are_isomorphic(SubParallelComputationGraph const &lhs, SubParallelComputationGraph const &rhs) {
  return find_isomorphism(lhs.raw_graph, rhs.raw_graph).has_value();
}

} // namespace FlexFlow

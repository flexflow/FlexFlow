#include "substitutions/sub_parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/create_lazy_copy_of_labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/with_labelling.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"

namespace FlexFlow {

std::unordered_set<parallel_layer_guid_t>
    get_parallel_layers(SubParallelComputationGraph const &sub_pcg) {
  return get_parallel_layers(pcg_from_sub_pcg_by_dropping_inputs(sub_pcg));
}

ParallelLayerAttrs
    get_parallel_layer_attrs(SubParallelComputationGraph const &spcg,
                             Node const &n) {
  return spcg.raw_graph.at(n);
}

PCGOperatorAttrs get_operator_attrs(SubParallelComputationGraph const &spcg,
                                    Node const &n) {
  return get_parallel_layer_attrs(spcg, n).op_attrs;
}

ParallelTensorAttrs
    get_parallel_tensor_attrs(SubParallelComputationGraph const &spcg,
                              OpenDataflowValue const &v) {
  return spcg.raw_graph.at(v);
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

SubParallelComputationGraph
    sub_pcg_from_partial_pcg(ParallelComputationGraph const &pcg,
                             std::unordered_set<Node> const &nodes) {
  auto as_open = view_as_labelled_open_dataflow_graph(pcg.raw_graph);
  OpenDataflowSubgraphResult subgraph_result = get_subgraph(as_open, nodes);
  return SubParallelComputationGraph{with_labelling(
      subgraph_result.graph,
      generate_map(nodes, [&](Node const &node) { return as_open.at(node); }),
      generate_map(get_open_dataflow_values(subgraph_result.graph),
                   [&](OpenDataflowValue const &value) {
                     if (value.has<DataflowGraphInput>()) {
                       return as_open.at(
                           subgraph_result.full_graph_values_to_subgraph_inputs
                               .at_r(value.get<DataflowGraphInput>()));
                     } else {
                       return as_open.at(value);
                     }
                   }))};
}

parallel_layer_guid_t
    get_parallel_layer_by_name(SubParallelComputationGraph const &pcg,
                               std::string const &name) {
  return get_parallel_layer_by_name(pcg_from_sub_pcg_by_dropping_inputs(pcg),
                                    name);
}

} // namespace FlexFlow

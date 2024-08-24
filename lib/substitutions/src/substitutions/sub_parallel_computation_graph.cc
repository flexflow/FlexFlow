#include "substitutions/sub_parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/values.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/create_lazy_copy_of_labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/from_labelled_open_dataflow_graph_data.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_incoming_edges.h"
#include "utils/graph/dataflow_graph/algorithms/get_outgoing_edges.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/as_dot.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_value_uses.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_input_edges.h"
#include "op-attrs/pcg_operator_attrs.h"

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

std::unordered_set<SubParallelComputationGraphEdge> get_incoming_edges(SubParallelComputationGraph const &spcg, std::unordered_set<parallel_layer_guid_t> const &layers) {
  std::unordered_map<Node, std::vector<OpenDataflowEdge>> found_edges = get_incoming_edges(spcg.raw_graph, transform(layers, [](parallel_layer_guid_t const &l) { return l.raw_graph_node; }));
  std::unordered_set<OpenDataflowEdge> all_found_edges = set_union(values(found_edges));
  return transform(all_found_edges, [](OpenDataflowEdge const &e) { return SubParallelComputationGraphEdge{e}; });
}

std::unordered_set<ParallelComputationGraphEdge> get_outgoing_edges(SubParallelComputationGraph const &spcg, 
                                                                    std::unordered_set<parallel_layer_guid_t> const &layers,
                                                                    IncludeInternalEdges include_internal_edges) {
  std::unordered_set<DataflowEdge> raw_edges = get_outgoing_edges(spcg.raw_graph, 
                                                                  transform(layers, [](parallel_layer_guid_t const &l) { return l.raw_graph_node; }),
                                                                  include_internal_edges);
  return transform(raw_edges, [](DataflowEdge const &e) { return ParallelComputationGraphEdge{e}; });
}

std::unordered_set<SubParallelComputationGraphEdge> get_subgraph_incoming_edges(SubParallelComputationGraph const &spcg,
                                                                                std::unordered_set<parallel_layer_guid_t> const &subgraph) {
  std::unordered_set<Node> raw_subgraph = transform(subgraph, [](parallel_layer_guid_t const &l) { return l.raw_graph_node; });
  std::unordered_set<OpenDataflowEdge> raw_incoming_edges = get_subgraph_incoming_edges(spcg.raw_graph, raw_subgraph);

  return transform(raw_incoming_edges, [](OpenDataflowEdge const &e) { return SubParallelComputationGraphEdge{e}; });
}

std::unordered_set<parallel_tensor_use_t> get_parallel_tensor_uses(SubParallelComputationGraph const &spcg,
                                                                   open_parallel_tensor_guid_t const &t) {
  std::unordered_set<DataflowInput> raw_uses = get_open_dataflow_value_uses(spcg.raw_graph, t.raw_open_dataflow_value);
  return transform(raw_uses, [](DataflowInput const &i) { return parallel_tensor_use_t{i}; });
}

SubParallelComputationGraphData get_sub_pcg_data(SubParallelComputationGraph const &pcg) {
  LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorAttrs> raw_data = get_graph_data(pcg.raw_graph);

  return SubParallelComputationGraphData{
    map_keys(raw_data.node_data, [](Node const &n) { return parallel_layer_guid_t{n}; }),
    transform(raw_data.edges, [](OpenDataflowEdge const &e) { return SubParallelComputationGraphEdge{e}; }),
    transform(raw_data.inputs, [](DataflowGraphInput const &i) { return input_parallel_tensor_guid_t{i}; }),
    map_keys(raw_data.value_data, [](OpenDataflowValue const &v) { return open_parallel_tensor_guid_t{v}; }),
  };
}

SubParallelComputationGraph sub_pcg_from_graph_data(SubParallelComputationGraphData const &data) {
  LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorAttrs> raw_data = LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorAttrs>{
    map_keys(data.node_data, [](parallel_layer_guid_t const &l) { return l.raw_graph_node; }),
    transform(data.edges, [](SubParallelComputationGraphEdge const &e) { return e.raw_edge; }),
    transform(data.inputs, [](input_parallel_tensor_guid_t const &i) { return i.raw_dataflow_graph_input; }),
    map_keys(data.value_data, [](open_parallel_tensor_guid_t const &t) { return t.raw_open_dataflow_value; }),
  };

  return SubParallelComputationGraph{
    from_labelled_open_dataflow_graph_data(raw_data),
  };
}

SubParallelComputationGraph without_layer_names(SubParallelComputationGraph const &spcg) {
  return SubParallelComputationGraph{
    rewrite_node_labels(spcg.raw_graph, 
                        [](Node const &n, ParallelLayerAttrs const &old_attrs) {
                          ParallelLayerAttrs new_attrs = old_attrs;
                          new_attrs.name = std::nullopt;
                          return new_attrs;
                        }),
  };
}

bool are_isomorphic(SubParallelComputationGraph const &lhs, SubParallelComputationGraph const &rhs) {
  return find_isomorphism(without_layer_names(lhs).raw_graph, without_layer_names(rhs).raw_graph).has_value();
}

std::string as_dot(SubParallelComputationGraph const &spcg) {
  std::function<std::string(ParallelLayerAttrs const &)> get_node_label = [](ParallelLayerAttrs const &a) -> std::string { 
    RecordFormatter r = as_dot(a.op_attrs);

    if (a.name.has_value()) {
      RecordFormatter rr;
      rr << "Name" << a.name.value();
      r << rr;
    }

    std::ostringstream oss;
    oss << r;
    return oss.str();
  };

  std::function<std::string(ParallelTensorAttrs const &)> get_input_label = [](ParallelTensorAttrs const &a) -> std::string {
    RecordFormatter r;

    r << fmt::to_string(a.shape);

    std::ostringstream oss;
    oss << r;
    return oss.str();
  };

  return as_dot(spcg.raw_graph, get_node_label, get_input_label);
}

void debug_print_dot(SubParallelComputationGraph const &spcg) {
  std::cout << as_dot(spcg) << std::endl;
}

} // namespace FlexFlow

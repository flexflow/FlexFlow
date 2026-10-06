#include "substitutions/sub_parallel_computation_graph.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/values.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_subgraph_outgoing_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/view_as_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/find_isomorphism_between_labelled_open_kwarg_dataflow_graphs.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/get_labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_view_as_dot.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_node_labels.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/view_from_labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_incoming_open_kwarg_dataflow_values_for_node.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_subgraph_incoming_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_value_uses.h"

namespace FlexFlow {

std::set<parallel_layer_guid_t>
    spcg_get_parallel_layers(SubParallelComputationGraph const &sub_pcg) {
  return transform(get_nodes(sub_pcg.raw_graph),
                   [](Node const &n) { return parallel_layer_guid_t{n}; });
}

std::set<open_parallel_tensor_guid_t>
    get_parallel_tensors(SubParallelComputationGraph const &sub_pcg) {
  return transform(get_all_open_kwarg_dataflow_values(sub_pcg.raw_graph),
                   [](OpenKwargDataflowValue<int, TensorSlotName> const &v)
                       -> open_parallel_tensor_guid_t {
                     return open_parallel_tensor_guid_t{v};
                   });
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
  return SubParallelComputationGraph(
      LabelledOpenKwargDataflowGraph<ParallelLayerAttrs,
                                     ParallelTensorAttrs,
                                     int,
                                     TensorSlotName>::
          create_copy_of<
              UnorderedSetLabelledOpenKwargDataflowGraph<ParallelLayerAttrs,
                                                         ParallelTensorAttrs,
                                                         int,
                                                         TensorSlotName>>(
              view_as_labelled_open_kwarg_dataflow_graph<ParallelLayerAttrs,
                                                         ParallelTensorAttrs,
                                                         int,
                                                         TensorSlotName>(
                  pcg.raw_graph)));
}

ParallelComputationGraph pcg_from_sub_pcg_by_dropping_inputs(
    SubParallelComputationGraph const &sub_pcg) {
  return ParallelComputationGraph{
      LabelledKwargDataflowGraph<ParallelLayerAttrs,
                                 ParallelTensorAttrs,
                                 TensorSlotName>::
          create_copy_of<
              UnorderedSetLabelledOpenKwargDataflowGraph<ParallelLayerAttrs,
                                                         ParallelTensorAttrs,
                                                         int,
                                                         TensorSlotName>>(
              sub_pcg.raw_graph)};
}

parallel_layer_guid_t
    get_parallel_layer_by_name(SubParallelComputationGraph const &pcg,
                               std::string const &name) {
  return get_parallel_layer_by_name(pcg_from_sub_pcg_by_dropping_inputs(pcg),
                                    name);
}

std::map<TensorSlotName, open_parallel_tensor_guid_t>
    get_layer_inputs(SubParallelComputationGraph const &pcg,
                     parallel_layer_guid_t const &layer) {
  return map_values(get_incoming_open_kwarg_dataflow_values_for_node(
                        pcg.raw_graph, layer.raw_graph_node),
                    [](OpenKwargDataflowValue<int, TensorSlotName> const &v) {
                      return open_parallel_tensor_guid_t{v};
                    });
}

std::map<TensorSlotName, parallel_tensor_guid_t>
    get_outgoing_tensors(SubParallelComputationGraph const &pcg,
                         parallel_layer_guid_t const &layer) {
  return map_values(get_outgoing_kwarg_dataflow_outputs_for_node(
                        pcg.raw_graph, layer.raw_graph_node),
                    [](KwargDataflowOutput<TensorSlotName> const &o) {
                      return parallel_tensor_guid_t{o};
                    });
}

std::set<ParallelComputationGraphEdge>
    get_subgraph_outgoing_edges(SubParallelComputationGraph const &spcg,
                                std::set<parallel_layer_guid_t> const &layers) {
  std::set<KwargDataflowEdge<TensorSlotName>> raw_edges =
      get_kwarg_dataflow_subgraph_outgoing_edges(
          spcg.raw_graph, transform(layers, [](parallel_layer_guid_t const &l) {
            return l.raw_graph_node;
          }));
  return transform(raw_edges, [](KwargDataflowEdge<TensorSlotName> const &e) {
    return ParallelComputationGraphEdge{e};
  });
}

std::set<SubParallelComputationGraphEdge> get_subgraph_incoming_edges(
    SubParallelComputationGraph const &spcg,
    std::set<parallel_layer_guid_t> const &subgraph) {
  std::set<Node> raw_subgraph =
      transform(subgraph, [](parallel_layer_guid_t const &l) {
        return l.raw_graph_node;
      });
  std::set<OpenKwargDataflowEdge<int, TensorSlotName>> raw_incoming_edges =
      get_open_kwarg_dataflow_subgraph_incoming_edges(spcg.raw_graph,
                                                      raw_subgraph);

  return transform(raw_incoming_edges,
                   [](OpenKwargDataflowEdge<int, TensorSlotName> const &e) {
                     return SubParallelComputationGraphEdge{e};
                   });
}

std::set<parallel_tensor_use_t>
    get_open_parallel_tensor_uses(SubParallelComputationGraph const &spcg,
                                  open_parallel_tensor_guid_t const &t) {
  std::set<KwargDataflowInput<TensorSlotName>> raw_uses =
      get_open_kwarg_dataflow_value_uses(spcg.raw_graph,
                                         t.raw_open_dataflow_value);
  return transform(raw_uses, [](KwargDataflowInput<TensorSlotName> const &i) {
    return parallel_tensor_use_t{i};
  });
}

SubParallelComputationGraphData
    get_sub_pcg_data(SubParallelComputationGraph const &pcg) {
  LabelledOpenKwargDataflowGraphData<ParallelLayerAttrs,
                                     ParallelTensorAttrs,
                                     int,
                                     TensorSlotName>
      raw_data = get_labelled_open_kwarg_dataflow_graph_data(pcg.raw_graph);

  require_labelled_open_kwarg_dataflow_graph_data_is_valid(raw_data);

  return SubParallelComputationGraphData{
      map_keys(raw_data.node_data,
               [](Node const &n) { return parallel_layer_guid_t{n}; }),
      transform(raw_data.edges,
                [](OpenKwargDataflowEdge<int, TensorSlotName> const &e) {
                  return SubParallelComputationGraphEdge{e};
                }),
      transform(raw_data.inputs,
                [](KwargDataflowGraphInput<int> const &i) {
                  return input_parallel_tensor_guid_t{i};
                }),
      map_keys(raw_data.value_data,
               [](OpenKwargDataflowValue<int, TensorSlotName> const &v) {
                 return open_parallel_tensor_guid_t{v};
               }),
  };
}

SubParallelComputationGraph
    sub_pcg_from_graph_data(SubParallelComputationGraphData const &data) {
  LabelledOpenKwargDataflowGraphData<ParallelLayerAttrs,
                                     ParallelTensorAttrs,
                                     int,
                                     TensorSlotName>
      raw_data = LabelledOpenKwargDataflowGraphData<ParallelLayerAttrs,
                                                    ParallelTensorAttrs,
                                                    int,
                                                    TensorSlotName>{
          map_keys(
              data.node_data,
              [](parallel_layer_guid_t const &l) { return l.raw_graph_node; }),
          transform(data.edges,
                    [](SubParallelComputationGraphEdge const &e) {
                      return e.raw_edge;
                    }),
          transform(data.inputs,
                    [](input_parallel_tensor_guid_t const &i) {
                      return i.raw_dataflow_graph_input;
                    }),
          map_keys(data.value_data,
                   [](open_parallel_tensor_guid_t const &t) {
                     return t.raw_open_dataflow_value;
                   }),
      };

  require_labelled_open_kwarg_dataflow_graph_data_is_valid(raw_data);

  return SubParallelComputationGraph{
      view_from_labelled_open_kwarg_dataflow_graph_data(raw_data),
  };
}

SubParallelComputationGraph
    without_layer_names(SubParallelComputationGraph const &spcg) {
  return SubParallelComputationGraph{
      rewrite_labelled_open_kwarg_dataflow_graph_node_labels(
          spcg.raw_graph,
          [](Node const &n, ParallelLayerAttrs const &old_attrs) {
            ParallelLayerAttrs new_attrs = old_attrs;
            new_attrs.name = std::nullopt;
            return new_attrs;
          }),
  };
}

bool sub_pcgs_are_isomorphic(SubParallelComputationGraph const &lhs,
                             SubParallelComputationGraph const &rhs) {
  return find_isomorphism_between_labelled_open_kwarg_dataflow_graphs(
             without_layer_names(lhs).raw_graph,
             without_layer_names(rhs).raw_graph)
      .has_value();
}

std::string sub_pcg_as_dot(SubParallelComputationGraph const &spcg) {
  std::function<nlohmann::json(ParallelLayerAttrs const &)> render_node_label =
      [](ParallelLayerAttrs const &a) -> nlohmann::json {
    nlohmann::json result = pcg_op_attrs_as_dot_json(a.op_attrs);

    if (a.name.has_value()) {
      result["Name"] = a.name.value();
    }

    return result;
  };

  std::function<nlohmann::json(ParallelTensorAttrs const &)>
      render_input_label = [](ParallelTensorAttrs const &a) -> nlohmann::json {
    RecordFormatter r = mk_empty_record(Orientation::HORIZONTAL);

    r << fmt::to_string(a.shape);

    std::ostringstream oss;
    oss << r;
    return oss.str();
  };

  std::function<nlohmann::json(TensorSlotName const &)> render_slot_name =
      [](TensorSlotName const &slot_name) -> nlohmann::json {
    return fmt::to_string(slot_name);
  };

  std::function<std::vector<TensorSlotName>(std::set<TensorSlotName> const &)>
      order_slots = [](std::set<TensorSlotName> const &slot_names)
      -> std::vector<TensorSlotName> { return sorted(slot_names); };

  return labelled_open_kwarg_dataflow_graph_view_as_dot(spcg.raw_graph,
                                                        render_node_label,
                                                        render_input_label,
                                                        render_slot_name,
                                                        order_slots);
}

void debug_print_dot(SubParallelComputationGraph const &spcg) {
  std::cout << sub_pcg_as_dot(spcg) << std::endl;
}

} // namespace FlexFlow

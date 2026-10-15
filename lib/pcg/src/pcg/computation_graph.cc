#include "pcg/computation_graph.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/shape_inference.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_element_counts.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/restrict_keys_strict.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/containers/zip_values_strict_with.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/get_subgraph_incoming_edges.h"
#include "utils/graph/dataflow_graph/algorithms/get_subgraph_outgoing_edges.h"
#include "utils/graph/digraph/algorithms/get_subgraph_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/find_isomorphism_between_kwarg_dataflow_graphs.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_incoming_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_subgraph_incoming_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_subgraph_outgoing_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_unused_kwarg_dataflow_values.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/rewrite_labelled_kwarg_dataflow_graph_node_labels.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/view_as_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/labelled_open_dataflow_graph_as_dot.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_view_as_dot.h"
#include "utils/graph/node/algorithms.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

ComputationGraph make_empty_computation_graph() {
  return ComputationGraph{
      LabelledKwargDataflowGraph<LayerAttrs, TensorAttrs, TensorSlotName>::
          create<UnorderedSetLabelledOpenKwargDataflowGraph<LayerAttrs,
                                                            TensorAttrs,
                                                            int,
                                                            TensorSlotName>>()};
}

std::set<layer_guid_t> get_layers(ComputationGraph const &cg) {
  return transform(get_nodes(cg.raw_graph),
                   [&](Node const &n) { return layer_guid_t{n}; });
}

LayerAddedResult
    add_layer(ComputationGraph &computation_graph,
              LayerAttrs const &layer_attrs,
              std::map<TensorSlotName, tensor_guid_t> const &inputs,
              std::map<TensorSlotName, tensor_guid_t> const &weights,
              std::optional<std::map<TensorSlotName, CreateGrad>> const
                  &maybe_output_flags) {

  std::map<TensorSlotName, TensorShape> input_shapes =
      map_values(inputs, [&](tensor_guid_t const &i) {
        return get_tensor_attrs(computation_graph, i).shape;
      });

  std::map<TensorSlotName, TensorShape> provided_weight_shapes =
      map_values(weights, [&](tensor_guid_t const &w) {
        return get_tensor_attrs(computation_graph, w).shape;
      });

  std::map<TensorSlotName, TensorShape> expected_weight_shapes =
      get_weight_shapes(layer_attrs.op_attrs, input_shapes);

  std::map<TensorSlotName, KwargDataflowOutput<TensorSlotName>> raw_inputs =
      map_values(inputs,
                 [&](tensor_guid_t const &t) { return t.raw_graph_output; });

  std::map<TensorSlotName, KwargDataflowOutput<TensorSlotName>> raw_weights =
      map_values(weights,
                 [&](tensor_guid_t const &t) { return t.raw_graph_output; });
  std::map<TensorSlotName, TensorShape> output_shapes =
      get_output_shapes(layer_attrs.op_attrs, input_shapes);

  std::map<TensorSlotName, CreateGrad> output_flags =
      maybe_output_flags.value_or(map_values(
          output_shapes, [&](TensorShape const &) { return CreateGrad::YES; }));

  std::map<TensorSlotName, TensorAttrs> output_attrs =
      zip_values_strict_with(output_shapes,
                             output_flags,
                             [](TensorShape const &shape,
                                CreateGrad const &create_grad) -> TensorAttrs {
                               return TensorAttrs{
                                   /*shape=*/shape,
                                   /*create_grad=*/create_grad,
                               };
                             });

  KwargNodeAddedResult<TensorSlotName> added =
      computation_graph.raw_graph.add_node(
          layer_attrs,
          binary_merge_disjoint_maps(raw_inputs, raw_weights),
          output_attrs);

  return LayerAddedResult{
      layer_guid_t{added.node},
      map_values(added.outputs,
                 [](KwargDataflowOutput<TensorSlotName> const &o) {
                   return tensor_guid_t{o};
                 }),
  };
}

LayerAddedResult add_input_layer(ComputationGraph &cg,
                                 TensorShape const &tensor_shape,
                                 CreateGrad create_grad) {
  LayerAttrs layer_attrs = LayerAttrs{
      /*op_attrs=*/ComputationGraphOpAttrs{InputAttrs{tensor_shape}},
      /*name=*/std::nullopt,
  };

  return add_layer(cg,
                   layer_attrs,
                   /*inputs=*/{},
                   /*weights=*/{},
                   /*outputs=*/
                   std::map<TensorSlotName, CreateGrad>{
                       {TensorSlotName::OUTPUT, create_grad},
                   });
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
  std::vector<Node> layers = reversed(get_topological_ordering(cg.raw_graph));
  return transform(
      layers, [&](Node const &e) -> layer_guid_t { return layer_guid_t{e}; });
}

std::map<TensorSlotName, tensor_guid_t>
    get_outgoing_tensors(ComputationGraph const &cg, layer_guid_t n) {
  return map_values(
      get_outgoing_kwarg_dataflow_outputs_for_node(cg.raw_graph, n.raw_node),
      [](KwargDataflowOutput<TensorSlotName> const &o) {
        return tensor_guid_t{o};
      });
}

std::map<TensorSlotName, tensor_guid_t>
    get_incoming_tensors(ComputationGraph const &cg, layer_guid_t n) {
  return map_values(
      get_incoming_kwarg_dataflow_outputs_for_node(cg.raw_graph, n.raw_node),
      [](KwargDataflowOutput<TensorSlotName> const &o) {
        return tensor_guid_t{o};
      });
}

std::map<TensorSlotName, TensorShape>
    get_incoming_input_shapes(ComputationGraph const &cg,
                              layer_guid_t const &n) {
  return map_values(get_incoming_inputs(cg, n), [&](tensor_guid_t const &t) {
    return get_tensor_attrs(cg, t).shape;
  });
}

static std::map<TensorSlotName, tensor_guid_t>
    get_incoming_tensors_with_role(ComputationGraph const &cg,
                                   layer_guid_t const &l,
                                   IncomingTensorRole desired_role) {
  ComputationGraphOpAttrs attrs = get_layer_attrs(cg, l).op_attrs;

  std::map<TensorSlotName, tensor_guid_t> incoming_tensors =
      get_incoming_tensors(cg, l);

  std::map<TensorSlotName, IncomingTensorRole> incoming_slot_roles =
      get_incoming_tensor_roles(attrs);

  ASSERT(incoming_tensors.size() == incoming_slot_roles.size());

  std::set<TensorSlotName> slots_with_desired_role =
      keys(filter_values(incoming_slot_roles, [&](IncomingTensorRole role) {
        return role == desired_role;
      }));

  return restrict_keys_strict(incoming_tensors, slots_with_desired_role);
}

std::map<TensorSlotName, tensor_guid_t>
    get_incoming_inputs(ComputationGraph const &cg, layer_guid_t const &l) {
  return get_incoming_tensors_with_role(cg, l, IncomingTensorRole::INPUT);
}

std::map<TensorSlotName, tensor_guid_t>
    get_incoming_weights(ComputationGraph const &cg, layer_guid_t const &l) {
  return get_incoming_tensors_with_role(cg, l, IncomingTensorRole::WEIGHT);
}

std::set<tensor_guid_t> get_all_tensors(ComputationGraph const &cg) {
  return transform(get_all_kwarg_dataflow_outputs(cg.raw_graph),
                   [](KwargDataflowOutput<TensorSlotName> const &t) {
                     return tensor_guid_t(t);
                   });
}

std::map<tensor_guid_t, TensorAttrs>
    get_all_tensor_attrs(ComputationGraph const &cg) {
  std::set<tensor_guid_t> all_tensors = get_all_tensors(cg);
  std::map<tensor_guid_t, TensorAttrs> all_tensor_attrs;
  for (tensor_guid_t const &tensor_guid : all_tensors) {
    all_tensor_attrs.insert({tensor_guid, get_tensor_attrs(cg, tensor_guid)});
  }
  return all_tensor_attrs;
}

std::set<tensor_guid_t> cg_get_unused_tensors(ComputationGraph const &cg) {
  return transform(
      get_unused_kwarg_dataflow_values(cg.raw_graph),
      [&](KwargDataflowOutput<TensorSlotName> const &o) -> tensor_guid_t {
        return tensor_guid_t{o};
      });
}

std::set<ComputationGraphEdge>
    get_subgraph_incoming_edges(ComputationGraph const &cg,
                                std::set<layer_guid_t> const &subgraph_nodes) {

  std::set<Node> raw_subgraph_nodes = transform(
      subgraph_nodes, [](layer_guid_t const &l) { return l.raw_node; });
  std::set<KwargDataflowEdge<TensorSlotName>> raw_incoming_edges =
      get_kwarg_dataflow_subgraph_incoming_edges(cg.raw_graph,
                                                 raw_subgraph_nodes);

  return transform(raw_incoming_edges,
                   [](KwargDataflowEdge<TensorSlotName> const &e) {
                     return ComputationGraphEdge{e};
                   });
}

std::set<ComputationGraphEdge>
    get_subgraph_outgoing_edges(ComputationGraph const &cg,
                                std::set<layer_guid_t> const &subgraph_nodes) {

  std::set<Node> raw_subgraph_nodes = transform(
      subgraph_nodes, [](layer_guid_t const &l) { return l.raw_node; });
  std::set<KwargDataflowEdge<TensorSlotName>> raw_outgoing_edges =
      get_kwarg_dataflow_subgraph_outgoing_edges(cg.raw_graph,
                                                 raw_subgraph_nodes);

  return transform(raw_outgoing_edges,
                   [](KwargDataflowEdge<TensorSlotName> const &e) {
                     return ComputationGraphEdge{e};
                   });
}

std::set<layer_guid_t>
    get_subgraph_successors(ComputationGraph const &cg,
                            std::set<layer_guid_t> const &subgraph_nodes) {

  std::set<Node> raw_subgraph_nodes = transform(
      subgraph_nodes, [](layer_guid_t const &l) { return l.raw_node; });
  std::set<Node> raw_successors =
      get_subgraph_successors(cg.raw_graph, raw_subgraph_nodes);

  return transform(raw_successors,
                   [](Node const &n) { return layer_guid_t{n}; });
}

LayerAttrs get_layer_attrs(ComputationGraph const &cg, layer_guid_t const &n) {
  return cg.raw_graph.at(n.raw_node);
}

std::map<layer_guid_t, LayerAttrs>
    get_layer_attrs_mapping(ComputationGraph const &cg) {
  std::map<layer_guid_t, LayerAttrs> layer_attrs_mapping;
  for (layer_guid_t const &layer_guid : get_layers(cg)) {
    layer_attrs_mapping.insert({layer_guid, get_layer_attrs(cg, layer_guid)});
  }
  return layer_attrs_mapping;
}

layer_guid_t get_layer_by_name(ComputationGraph const &cg,
                               std::string const &name) {
  std::set<layer_guid_t> found =
      filter(get_layers(cg), [&](layer_guid_t const &l) {
        return get_layer_attrs(cg, l).name == name;
      });
  return get_only(found);
}

ComputationGraph without_layer_names(ComputationGraph const &cg) {
  LabelledKwargDataflowGraphView<LayerAttrs, TensorAttrs, TensorSlotName>
      relabelled = rewrite_labelled_kwarg_dataflow_graph_node_labels(
          cg.raw_graph,
          [](Node const &n, LayerAttrs const &old_attrs) -> LayerAttrs {
            LayerAttrs new_attrs = old_attrs;
            new_attrs.name = std::nullopt;
            return new_attrs;
          });
  return ComputationGraph{
      LabelledKwargDataflowGraph<LayerAttrs, TensorAttrs, TensorSlotName>::
          create_copy_of<
              UnorderedSetLabelledOpenKwargDataflowGraph<LayerAttrs,
                                                         TensorAttrs,
                                                         int,
                                                         TensorSlotName>>(
              relabelled),
  };
}

std::map<OperatorType, positive_int>
    operator_type_counts_in_computation_graph(ComputationGraph const &cg) {
  std::vector<layer_guid_t> layers = vector_of(get_layers(cg));
  std::vector<OperatorType> operator_types =
      transform(layers, [&](layer_guid_t const &l) -> OperatorType {
        return get_op_type(get_layer_attrs(cg, l).op_attrs);
      });

  return get_element_counts(operator_types);
}

bool computation_graphs_are_isomorphic(ComputationGraph const &lhs,
                                       ComputationGraph const &rhs) {
  return find_isomorphism_between_kwarg_dataflow_graphs(
             without_layer_names(lhs).raw_graph,
             without_layer_names(rhs).raw_graph)
      .has_value();
}

std::string as_dot(ComputationGraph const &cg) {
  std::function<nlohmann::json(LayerAttrs const &)> get_node_label =
      [](LayerAttrs const &a) -> nlohmann::json {
    nlohmann::json result = a;

    return result;
  };

  std::function<nlohmann::json(TensorAttrs const &)> get_input_label =
      [](TensorAttrs const &a) -> nlohmann::json {
    nlohmann::json result = a;

    return result;
  };

  std::function<nlohmann::json(TensorSlotName const &)> render_slot_name =
      [](TensorSlotName const &s) -> nlohmann::json {
    nlohmann::json result = fmt::to_string(s);

    return result;
  };

  std::function<std::vector<TensorSlotName>(std::set<TensorSlotName> const &)>
      order_slots =
          [](std::set<TensorSlotName> const &unordered) -> nlohmann::json {
    return sorted(unordered);
  };

  return labelled_open_kwarg_dataflow_graph_view_as_dot(
      view_as_labelled_open_kwarg_dataflow_graph<LayerAttrs,
                                                 TensorAttrs,
                                                 int,
                                                 TensorSlotName>(cg.raw_graph),
      get_node_label,
      get_input_label,
      render_slot_name,
      order_slots);
}

void debug_print_dot(ComputationGraph const &cg) {
  std::cerr << as_dot(cg) << std::endl;
}

} // namespace FlexFlow

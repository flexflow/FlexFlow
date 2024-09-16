#include "pcg/computation_graph.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/get_subgraph_incoming_edges.h"
#include "utils/graph/dataflow_graph/algorithms/get_subgraph_outgoing_edges.h"
#include "utils/graph/digraph/algorithms/get_subgraph_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/as_dot.h"
#include "utils/graph/node/algorithms.h"
#include "utils/record_formatter.h"

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
  std::vector<DataflowOutput> raw_inputs = transform(
      inputs, [](tensor_guid_t const &t) { return t.raw_graph_output; });

  NodeAddedResult added =
      computation_graph.raw_graph.add_node(attrs, raw_inputs, outputs);

  return LayerAddedResult{
      layer_guid_t{added.node},
      transform(added.outputs,
                [](DataflowOutput const &o) { return tensor_guid_t{o}; }),
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
  std::vector<Node> layers = reversed(get_topological_ordering(cg.raw_graph));
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

static std::vector<tensor_guid_t>
    get_incoming_tensors_with_role(ComputationGraph const &cg,
                                   layer_guid_t const &l,
                                   IncomingTensorRole desired_role) {
  ComputationGraphOpAttrs attrs = get_layer_attrs(cg, l).attrs;

  std::vector<tensor_guid_t> incoming_tensors = get_incoming_tensors(cg, l);

  std::vector<IncomingTensorRole> incoming_tensor_roles =
      get_incoming_tensor_roles(attrs, incoming_tensors.size());

  assert(incoming_tensors.size() == incoming_tensor_roles.size());

  std::vector<tensor_guid_t> result =
      filtrans(zip(incoming_tensors, incoming_tensor_roles),
               [&](std::pair<tensor_guid_t, IncomingTensorRole> const &p)
                   -> std::optional<tensor_guid_t> {
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

std::vector<tensor_guid_t> get_incoming_inputs(ComputationGraph const &cg,
                                               layer_guid_t const &l) {
  return get_incoming_tensors_with_role(cg, l, IncomingTensorRole::INPUT);
}

std::vector<tensor_guid_t> get_incoming_weights(ComputationGraph const &cg,
                                                layer_guid_t const &l) {
  return get_incoming_tensors_with_role(cg, l, IncomingTensorRole::WEIGHT);
}



std::unordered_set<ComputationGraphEdge> get_subgraph_incoming_edges(
    ComputationGraph const &cg,
    std::unordered_set<layer_guid_t> const &subgraph_nodes) {

  std::unordered_set<Node> raw_subgraph_nodes = transform(
      subgraph_nodes, [](layer_guid_t const &l) { return l.raw_node; });
  std::unordered_set<DataflowEdge> raw_incoming_edges =
      get_subgraph_incoming_edges(cg.raw_graph, raw_subgraph_nodes);

  return transform(raw_incoming_edges, [](DataflowEdge const &e) {
    return ComputationGraphEdge{e};
  });
}

std::unordered_set<ComputationGraphEdge> get_subgraph_outgoing_edges(
    ComputationGraph const &cg,
    std::unordered_set<layer_guid_t> const &subgraph_nodes) {

  std::unordered_set<Node> raw_subgraph_nodes = transform(
      subgraph_nodes, [](layer_guid_t const &l) { return l.raw_node; });
  std::unordered_set<DataflowEdge> raw_outgoing_edges =
      get_subgraph_outgoing_edges(cg.raw_graph, raw_subgraph_nodes);

  return transform(raw_outgoing_edges, [](DataflowEdge const &e) {
    return ComputationGraphEdge{e};
  });
}

std::unordered_set<layer_guid_t> get_subgraph_successors(
    ComputationGraph const &cg,
    std::unordered_set<layer_guid_t> const &subgraph_nodes) {

  std::unordered_set<Node> raw_subgraph_nodes = transform(
      subgraph_nodes, [](layer_guid_t const &l) { return l.raw_node; });
  std::unordered_set<Node> raw_successors =
      get_subgraph_successors(cg.raw_graph, raw_subgraph_nodes);

  return transform(raw_successors,
                   [](Node const &n) { return layer_guid_t{n}; });
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

std::string as_dot(ComputationGraph const &cg) {
  std::function<std::string(LayerAttrs const &)> get_node_label =
      [](LayerAttrs const &a) -> std::string {
    RecordFormatter r = as_dot(a.attrs);

    if (a.name.has_value()) {
      RecordFormatter rr;
      rr << "Name" << a.name.value();
      r << rr;
    }

    std::ostringstream oss;
    oss << r;
    return oss.str();
  };

  std::function<std::string(TensorAttrs const &)> get_input_label =
      [](TensorAttrs const &a) -> std::string {
    RecordFormatter r;

    r << fmt::to_string(a.shape);

    std::ostringstream oss;
    oss << r;
    return oss.str();
  };

  return as_dot(view_as_labelled_open_dataflow_graph(cg.raw_graph),
                get_node_label,
                get_input_label);
}

void debug_print_dot(ComputationGraph const &cg) {
  std::cout << as_dot(cg) << std::endl;
}

} // namespace FlexFlow

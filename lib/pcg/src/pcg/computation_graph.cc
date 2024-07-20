#include "pcg/computation_graph.h"
#include "utils/containers.h"

namespace FlexFlow {

ComputationGraph make_empty_computation_graph() {
  return ComputationGraph{DataflowGraph<LayerAttrs, TensorAttrs>{}};
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
  std::vector<Node> layers =
      get_topological_ordering(cg.raw_graph.get_raw_graph());
  return transform(
      layers, [&](Node const &e) -> layer_guid_t { return layer_guid_t{e}; });
}

static std::vector<tensor_guid_t>
    sort_edge_set(std::unordered_set<MultiDiEdge> const &edges) {
  return transform(
      sorted_by(edges, compare_by<MultiDiEdge>([](MultiDiEdge const &e) {
                  return e.src_idx;
                })),
      [&](MultiDiEdge const &e) -> tensor_guid_t { return tensor_guid_t{e}; });
}

std::vector<tensor_guid_t> get_outgoing_tensors(ComputationGraph const &cg,
                                                layer_guid_t n) {
  return transform(cg.raw_graph.get_output_map().at(n.raw_node),
                   [&](MultiDiOutput const &o) -> tensor_guid_t {
                     return tensor_guid_t{o};
                   });
}

std::vector<tensor_guid_t> get_incoming_tensors(ComputationGraph const &cg,
                                                layer_guid_t n) {
  return sort_edge_set(
      get_incoming_edges(cg.raw_graph.get_raw_graph(), n.raw_node));
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

#include "utils/graph/open_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/get_all_permutations.h"
#include "utils/containers/get_first.h"
#include "utils/containers/keys.h"
#include "utils/containers/values.h"
#include "utils/containers/zip.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/algorithms/new_node.dtg.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_incoming_edge.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_source_nodes.h"
#include "utils/graph/open_dataflow_graph/algorithms/is_isomorphic_under.h"
#include "utils/graph/open_dataflow_graph/algorithms/new_dataflow_graph_input.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include <queue>
#include "utils/containers/is_subseteq_of.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_incoming_edges.h"

namespace FlexFlow {

static std::optional<OpenDataflowGraphIsomorphism> find_isomorphism_under_sink_node_mapping(OpenDataflowGraphView const &src_g,
                                                                                            OpenDataflowGraphView const &dst_g,
                                                                                            bidict<Node, Node> const &sink_node_mapping) {
  {
    std::unordered_set<Node> already_mapped_src_nodes = left_entries(sink_node_mapping);
    std::unordered_set<Node> src_g_sink_nodes = get_sinks(src_g);
    assert (already_mapped_src_nodes == src_g_sink_nodes);
  }

  { 
    std::unordered_set<Node> already_mapped_dst_nodes = right_entries(sink_node_mapping);
    std::unordered_set<Node> dst_g_sink_nodes = get_sinks(dst_g);
    assert (already_mapped_dst_nodes == dst_g_sink_nodes);
  }

  std::optional<OpenDataflowGraphIsomorphism> result = OpenDataflowGraphIsomorphism{
    {},
    {},
  };

  auto fail = [&]() -> void {
    result = std::nullopt;
  };

  auto has_failed = [&]() -> bool {
    return result == std::nullopt;
  };

  std::function<void(Node const &, Node const &)> unify_nodes;
  std::function<void(OpenDataflowEdge const &, OpenDataflowEdge const &)> unify_edges;
  std::function<void(DataflowGraphInput const &, DataflowGraphInput const &)> unify_graph_inputs;
  std::function<void(OpenDataflowValue const &, OpenDataflowValue const &)> unify_values;
  std::function<void(DataflowOutput const &, DataflowOutput const &)> unify_outputs;

  unify_outputs = [&](DataflowOutput const &src_output, DataflowOutput const &dst_output) {
    if (has_failed()) {
      return;
    }

    if (src_output.idx != dst_output.idx) {
      result = std::nullopt;
      return;
    }

    unify_nodes(src_output.node, dst_output.node);
  };

  unify_values = [&](OpenDataflowValue const &src_val, OpenDataflowValue const &dst_val) {
    if (has_failed()) {
      return;
    }

    if (src_val.index() != dst_val.index()) {
      fail();
      return;
    }

    if (src_val.has<DataflowOutput>()) {
      unify_outputs(src_val.get<DataflowOutput>(), dst_val.get<DataflowOutput>()); 
    } else {
      unify_graph_inputs(src_val.get<DataflowGraphInput>(), dst_val.get<DataflowGraphInput>());
    }
  };

  unify_graph_inputs = [&](DataflowGraphInput const &src, DataflowGraphInput const &dst) {
    if (has_failed()) {
      return;
    }

    if (result->input_mapping.contains_l(src) && result->input_mapping.at_l(src) != dst) {
      fail();
      return;
    }
    if (result->input_mapping.contains_r(dst) && result->input_mapping.at_r(dst) != src) {
      fail();
      return;
    }

    result->input_mapping.equate(src, dst);
  };

  unify_edges = [&](OpenDataflowEdge const &src_edge, OpenDataflowEdge const &dst_edge) {
    if (has_failed()) {
      return;
    }

    assert (get_open_dataflow_edge_dst(src_edge).idx == get_open_dataflow_edge_dst(dst_edge).idx);
    assert (get_open_dataflow_edge_dst(src_edge).node == result->node_mapping.at_r(get_open_dataflow_edge_dst(dst_edge).node));

    unify_values(get_open_dataflow_edge_source(src_edge), get_open_dataflow_edge_source(dst_edge));
  };

  unify_nodes = [&](Node const &src_node, Node const &dst_node) {
    if (has_failed()) {
      return;
    }

    if (result->node_mapping.contains(src_node, dst_node)) {
      return;
    }

    if (result->node_mapping.contains_l(src_node) && result->node_mapping.at_l(src_node) != dst_node) {
      fail();
      return;
    }
    if (result->node_mapping.contains_r(dst_node) && result->node_mapping.at_r(dst_node) != src_node) {
      fail();
      return;
    }

    result->node_mapping.equate(src_node, dst_node);

    std::vector<OpenDataflowEdge> src_incoming_edges = get_incoming_edges(src_g, src_node);
    std::vector<OpenDataflowEdge> dst_incoming_edges = get_incoming_edges(dst_g, dst_node);

    if (src_incoming_edges.size() != dst_incoming_edges.size()) {
      fail();
      return;
    }

    for (auto const &[src_edge, dst_edge] : zip(src_incoming_edges, dst_incoming_edges)) {
      unify_edges(src_edge, dst_edge);
    }
  };

  for (auto const &[src_node, dst_node] : sink_node_mapping) {
    unify_nodes(src_node, dst_node);
  }

  return result;
}

std::unordered_set<OpenDataflowGraphIsomorphism> find_isomorphisms(OpenDataflowGraphView const &src, OpenDataflowGraphView const &dst) {
  std::unordered_set<OpenDataflowGraphIsomorphism> result;

  std::vector<Node> src_sink_nodes = as_vector(get_sinks(src));
  
  for (std::vector<Node> const &dst_sink_nodes : get_all_permutations(get_sinks(dst))) {
    bidict<Node, Node> sink_node_mapping = bidict_from_keys_and_values(src_sink_nodes, dst_sink_nodes);
    std::optional<OpenDataflowGraphIsomorphism> found = find_isomorphism_under_sink_node_mapping(src, dst, sink_node_mapping);
    
    if (found.has_value()) {
      assert (is_isomorphic_under(src, dst, found.value()));

      result.insert(found.value());
    }
  }

  return result;
}

std::optional<OpenDataflowGraphIsomorphism> find_isomorphism(OpenDataflowGraphView const &src,
                                                             OpenDataflowGraphView const &dst) {
  std::unordered_set<OpenDataflowGraphIsomorphism> all_isomorphisms = find_isomorphisms(src, dst);

  if (all_isomorphisms.empty()) {
    return std::nullopt;
  } else {
    return get_first(all_isomorphisms);
  }
}

} // namespace FlexFlow

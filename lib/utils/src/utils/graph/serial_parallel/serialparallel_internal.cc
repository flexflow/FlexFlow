#include "utils/graph/serial_parallel/serialparallel_internal.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/extend.h"
#include "utils/containers/get_first.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_dominators.h"
#include "utils/graph/digraph/algorithms/get_imm_post_dominator.h"
#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "utils/graph/digraph/algorithms/inverse_line_graph/get_inverse_line_graph.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/multidigraph/algorithms/get_incoming_edges.h"
#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/parallel_reduction.h"
#include "utils/graph/serial_parallel/series_reduction.h"
#include "utils/graph/serial_parallel/sink_settings.dtg.h"
#include "utils/graph/serial_parallel/source_settings.dtg.h"

namespace FlexFlow {

Node find_source_node(DiGraphView const &g) {
  std::unordered_set<Node> srcs = get_sources(g);
  return get_only(srcs);
}

Node find_sink_node(DiGraphView const &g) {
  std::unordered_set<Node> sinks = get_sinks(g);
  return get_only(sinks);
}

std::optional<Node> find_bottleneck_node(DiGraphView const &g) {
  std::unordered_set<Node> sources = get_sources(g);
  std::unordered_set<Node> sinks = get_sinks(g);

  std::optional<Node> maybe_bottleneck = get_imm_post_dominator(g, sources);
  if (maybe_bottleneck.has_value()) {
    assert(contains(get_dominators(g, sinks), maybe_bottleneck.value()));
  }
  return maybe_bottleneck;
}

std::unordered_set<Node> from_source_to_sink(DiGraphView const &g,
                                             Node const &src,
                                             Node const &sink) {
  assert(contains(get_dominators(g, sink), src));

  std::vector<Node> bfs = get_bfs_ordering(g, {src});
  auto end = find(bfs, sink);
  assert(end != bfs.end());

  std::unordered_set<Node> result(bfs.cbegin(), ++end);
  return result;
}

std::unordered_set<Node>
    from_source_to_sink(DiGraphView const &g,
                        std::unordered_set<Node> const &srcs,
                        std::unordered_set<Node> const &sinks,
                        SourceSettings include_src,
                        SinkSettings include_sink) {
  assert(is_acyclic(g));

  Node contracted_src = get_first(srcs);
  Node contracted_sink = get_first(sinks);
  std::unordered_map<Node, Node> contraction;
  for (Node const &src : srcs) {
    contraction.insert({src, contracted_src});
  }
  for (Node const &sink : sinks) {
    contraction.insert({sink, contracted_sink});
  }
  DiGraphView contracted_view = apply_contraction(g, contraction);

  std::unordered_set<Node> result =
      from_source_to_sink(contracted_view, contracted_src, contracted_sink);

  assert(contains(result, contracted_src));
  assert(contains(result, contracted_sink));

  result.erase(contracted_src);
  result.erase(contracted_sink);

  assert(are_disjoint(result, srcs));
  assert(are_disjoint(result, sinks));

  if (include_src == SourceSettings::INCLUDE_SOURCE_NODES) {
    result = set_union(result, srcs);
  }
  if (include_sink == SinkSettings::INCLUDE_SINK_NODES) {
    result = set_union(result, sinks);
  }
  return result;
}

DiGraphView source_to_sink_subgraph(DiGraphView const &g,
                                    std::unordered_set<Node> const &srcs,
                                    std::unordered_set<Node> const &sinks,
                                    SourceSettings include_src,
                                    SinkSettings include_sink) {
  return get_subgraph(
      g, from_source_to_sink(g, srcs, sinks, include_src, include_sink));
}

std::optional<std::variant<IntermediateSpDecompositionTree, Node>>
    sp_decomposition(DiGraphView const &g) {
  InverseLineGraphResult inverse_line_graph_result = get_inverse_line_graph(g);
  MultiDiGraph ttsp = MultiDiGraph::materialize_copy_of<AdjacencyMultiDiGraph>(
      inverse_line_graph_result.graph);
  std::unordered_map<MultiDiEdge,
                     std::variant<IntermediateSpDecompositionTree, Node>>
      ttsp_edge_to_sp_tree = map_values(
          inverse_line_graph_result.inverse_edge_to_line_node_bidict
              .as_unordered_map(),
          [](Node const &n) {
            return std::variant<IntermediateSpDecompositionTree, Node>{n};
          });

  while (true) {
    assert(ttsp_edge_to_sp_tree.size() == get_edges(ttsp).size());
    std::optional<ParallelReduction> maybe_parallel_reduction =
        find_parallel_reduction(ttsp);
    if (maybe_parallel_reduction.has_value()) {
      ParallelReduction parallel_reduction = maybe_parallel_reduction.value();
      auto [e1, e2] = parallel_reduction.edges.ordered();
      MultiDiEdge merged = apply_parallel_reduction(ttsp, parallel_reduction);
      std::variant<IntermediateSpDecompositionTree, Node> new_tree =
          IntermediateSpDecompositionTree{
              SplitType::PARALLEL,
              {ttsp_edge_to_sp_tree.at(e1), ttsp_edge_to_sp_tree.at(e2)},
          };
      ttsp_edge_to_sp_tree.erase(e1);
      ttsp_edge_to_sp_tree.erase(e2);
      ttsp_edge_to_sp_tree.insert({merged, new_tree});

      continue;
    }

    std::optional<SeriesReduction> maybe_series_reduction =
        find_series_reduction(ttsp);
    if (maybe_series_reduction.has_value()) {
      SeriesReduction series_reduction = maybe_series_reduction.value();
      MultiDiEdge e1 = series_reduction.first;
      MultiDiEdge e2 = series_reduction.second;
      MultiDiEdge merged = apply_series_reduction(ttsp, series_reduction);
      std::variant<IntermediateSpDecompositionTree, Node> new_tree =
          IntermediateSpDecompositionTree{
              SplitType::SERIAL,
              {ttsp_edge_to_sp_tree.at(e1), ttsp_edge_to_sp_tree.at(e2)},
          };
      ttsp_edge_to_sp_tree.erase(e1);
      ttsp_edge_to_sp_tree.erase(e2);
      ttsp_edge_to_sp_tree.insert({merged, new_tree});
      continue;
    }

    if (get_nodes(ttsp).size() != 2) {
      return std::nullopt;
    }
    if (get_edges(ttsp).size() != 1) {
      return std::nullopt;
    }

    MultiDiEdge e = get_only(get_edges(ttsp));
    if (ttsp.get_multidiedge_src(e) != ttsp.get_multidiedge_dst(e)) {
      return ttsp_edge_to_sp_tree.at(e);
    }
  }
}

// IntermediateSpDecompositionTree parallel_decomposition(DiGraphView const &g)
// {
//   std::unordered_set<std::unordered_set<Node>> weakly_connected_components =
//       get_weakly_connected_components(g);
//   assert(weakly_connected_components.size() > 1);
//
//   IntermediateSpDecompositionTree split(SplitType::PARALLEL, {});
//   for (auto const &component : weakly_connected_components) {
//     split.children.push_back(sp_decomposition(get_subgraph(g, component)));
//   }
//
//   return split;
// }

} // namespace FlexFlow

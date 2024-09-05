#include "utils/graph/serial_parallel/get_serial_parallel_decomposition.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms/inverse_line_graph/get_inverse_line_graph.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/nary_sp_tree_from_binary.h"
#include "utils/graph/serial_parallel/parallel_reduction.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/series_reduction.h"

namespace FlexFlow {

std::optional<SerialParallelDecomposition>
    get_serial_parallel_decomposition(DiGraphView const &x) {

  DiGraphView transitively_reduced = transitive_reduction(x);
  // DiGraphView transitively_reduced = x;
  InverseLineGraphResult inverse_line_graph_result = ({
    std::optional<InverseLineGraphResult> maybe_line_graph =
        get_inverse_line_graph(transitively_reduced);
    if (!maybe_line_graph.has_value()) {
      return std::nullopt;
    }

    maybe_line_graph.value();
  });

  MultiDiGraph ttsp = MultiDiGraph::materialize_copy_of<AdjacencyMultiDiGraph>(
      inverse_line_graph_result.graph);
  std::unordered_map<MultiDiEdge,
                     BinarySPDecompositionTree>
      ttsp_edge_to_sp_tree = map_values(
          inverse_line_graph_result.inverse_edge_to_line_node_bidict
              .as_unordered_map(),
          [](Node const &n) {
            return BinarySPDecompositionTree{n};
          });

  while (true) {
    assert(ttsp_edge_to_sp_tree.size() == get_edges(ttsp).size());
    std::optional<ParallelReduction> maybe_parallel_reduction =
        find_parallel_reduction(ttsp);
    if (maybe_parallel_reduction.has_value()) {
      ParallelReduction parallel_reduction = maybe_parallel_reduction.value();
      auto [e1, e2] = parallel_reduction.edges.ordered();
      MultiDiEdge merged = apply_parallel_reduction(ttsp, parallel_reduction);
      BinarySPDecompositionTree new_tree =
          BinarySPDecompositionTree{
            BinaryParallelSplit{
              ttsp_edge_to_sp_tree.at(e1), 
              ttsp_edge_to_sp_tree.at(e2),
            },
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
      BinarySPDecompositionTree new_tree =
          BinarySPDecompositionTree{
            BinarySeriesSplit{
              ttsp_edge_to_sp_tree.at(e1), 
              ttsp_edge_to_sp_tree.at(e2),
            },
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
      return nary_sp_tree_from_binary(ttsp_edge_to_sp_tree.at(e));
    }
  }
}

} // namespace FlexFlow

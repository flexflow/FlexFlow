#include "utils/graph/serial_parallel/sp_ization/critical_path_preserving_sp_ization.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/algorithms/is_2_terminal_dag.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/normalize_sp_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/variant.h"

namespace FlexFlow {

static SerialSplit cut_off_head(SerialSplit const &s) {
  assert(s.children.size() > 0);
  return SerialSplit{std::vector<std::variant<ParallelSplit, Node>>(
      s.children.begin() + 1, s.children.end())};
}

/* Performs a parallel composition with coalescing, where components with a
 * common starting child are merged together
 * Example: to parallel compose S(1, 2, 5), S(1, 3, 4):
 *  without coalescing: P(S(1, 2, 5), S(1, 3, 4))
 *  with coalescing: S(1, P( S(2,5), S(3,4) ))
 */
static SerialParallelDecomposition parallel_composition_with_coalescing(
    std::unordered_set<SerialSplit> const &strands) {
  if (strands.size() == 1) {
    return SerialParallelDecomposition(get_only(strands));
  }

  // group strands by their first ("head") node
  std::unordered_map<std::variant<ParallelSplit, Node>,
                     std::unordered_set<SerialSplit>>
      grouped_strands;
  for (SerialSplit predecessor : filter(strands, [](SerialSplit const &serial) {
         return !is_empty(serial);
       })) {
    grouped_strands[predecessor.children.at(0)].insert(
        cut_off_head(predecessor));
  }

  // recursively coalesce the strands
  std::unordered_set<SerialParallelDecomposition> coalesced_strands;
  for (auto const &[head, tails] : grouped_strands) {
    SerialParallelDecomposition parallel_comp =
        parallel_composition_with_coalescing(tails);
    coalesced_strands.insert(serial_composition(
        {widen<SerialParallelDecomposition>(head), parallel_comp}));
  }

  return normalize_sp_decomposition(parallel_composition(coalesced_strands));
}

static SerialParallelDecomposition
    critical_path_preserving_sp_ization_unchecked_with_coalescing(
        DiGraphView const &g) {
  std::unordered_map<Node, SerialSplit> node_to_sp;

  Node source = get_only(get_sources(g));
  node_to_sp[source] = SerialSplit{{source}};

  for (Node const &node : get_topological_ordering(g)) {
    if (node == source) {
      continue;
    }
    std::unordered_set<SerialSplit> predecessors_as_sp =
        transform(get_predecessors(g, node),
                  [&](Node const &p) { return node_to_sp.at(p); });

    SerialParallelDecomposition parallel_composed_predecessors =
        parallel_composition_with_coalescing(predecessors_as_sp);
    SerialParallelDecomposition sp_decomp = serial_composition(
        {parallel_composed_predecessors, SerialParallelDecomposition(node)});
    node_to_sp[node] = sp_decomp.get<SerialSplit>();
  }

  Node sink = get_only(get_sinks(g));
  return normalize_sp_decomposition(
      SerialParallelDecomposition(node_to_sp.at(sink)));
}

SerialParallelDecomposition
    critical_path_preserving_sp_ization_with_coalescing(DiGraphView const &g) {
  assert(is_2_terminal_dag(g));
  return critical_path_preserving_sp_ization_unchecked_with_coalescing(g);
}

static SerialParallelDecomposition
    critical_path_preserving_sp_ization_unchecked(DiGraphView const &g) {
  std::unordered_map<Node, SerialParallelDecomposition> node_to_sp;

  for (Node const &node : get_topological_ordering(g)) {

    std::unordered_set<SerialParallelDecomposition> predecessors_as_sp =
        transform(get_predecessors(g, node),
                  [&](Node const &p) { return node_to_sp.at(p); });

    SerialParallelDecomposition sp_decomp = serial_composition(
        {normalize_sp_decomposition(parallel_composition(predecessors_as_sp)),
         SerialParallelDecomposition(node)});

    node_to_sp.emplace(node, sp_decomp);
  }

  Node sink = get_only(get_sinks(g));
  return node_to_sp.at(sink);
}

SerialParallelDecomposition
    critical_path_preserving_sp_ization(DiGraphView const &g) {
  assert(is_2_terminal_dag(g));
  return critical_path_preserving_sp_ization_unchecked(g);
}

} // namespace FlexFlow

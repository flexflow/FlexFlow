#include "utils/graph/serial_parallel/sp_ization/critical_path_preserving_sp_ization.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/algorithms/is_2_terminal_dag.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/parallel_composition.h"
#include "utils/graph/serial_parallel/serial_composition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_normalize.h"
#include "utils/variant.h"

namespace FlexFlow {

static SerialSplit cut_off_head(SerialSplit s) {
  assert(s.children.size() > 0);
  return SerialSplit{std::vector<std::variant<ParallelSplit, Node>>(
      s.children.begin() + 1, s.children.end())};
}

static SerialParallelDecomposition parallel_composition_with_coalescing(
    std::vector<SerialSplit> sp_predecessors) {
  if (sp_predecessors.size() == 1) {
    return SerialParallelDecomposition(get_only(sp_predecessors));
  }
  std::unordered_map<std::variant<ParallelSplit, Node>,
                     std::vector<SerialSplit>>
      coalescable;
  for (SerialSplit predecessor : sp_predecessors) {
    if (predecessor.children.size() == 0) {
      continue;
    } else {
      coalescable[predecessor.children[0]].push_back(predecessor);
    }
  }

  std::unordered_set<SerialParallelDecomposition> sp;
  for (auto const &item : coalescable) {
    std::variant<ParallelSplit, Node> head = item.first;
    std::vector<SerialSplit> sp_branches = item.second;
    std::vector<SerialSplit> cut_off = transform(sp_branches, cut_off_head);
    auto p_comp = parallel_composition_with_coalescing(cut_off);
    sp.insert(
        serial_composition({widen<SerialParallelDecomposition>(head), p_comp}));
  }
  return parallel_composition(sp);
}

static SerialParallelDecomposition
    critical_path_preserving_sp_ization_unchecked_with_coalescing(
        DiGraphView const &g) {
  std::unordered_map<Node, SerialSplit> node_to_sp;

  Node source = get_only(get_sources(g));
  node_to_sp[source] = SerialSplit{{source}}; // base-case

  for (Node const &node : get_topological_ordering(g)) {
    if (node == source) {
      continue;
    }
    std::vector<SerialSplit> sp_predecessors;
    for (Node const &p : get_predecessors(g, node)) {
      sp_predecessors.push_back(node_to_sp.at(p));
    }
    SerialParallelDecomposition parallel_composed_predecessors =
        parallel_composition_with_coalescing(sp_predecessors);
    SerialParallelDecomposition sp_decomp = serial_composition(
        {parallel_composed_predecessors, SerialParallelDecomposition(node)});
    assert(sp_decomp.has<SerialSplit>());
    node_to_sp[node] = sp_decomp.get<SerialSplit>();
  }

  Node sink = get_only(get_sinks(g));
  return normalize(SerialParallelDecomposition(node_to_sp.at(sink)));
}

SerialParallelDecomposition
    critical_path_preserving_sp_ization_with_coalescing(DiGraphView const &g) {
  assert(is_2_terminal_dag(g));
  return critical_path_preserving_sp_ization_unchecked_with_coalescing(g);
}

static SerialParallelDecomposition
    critical_path_preserving_sp_ization_unchecked(DiGraphView const &g) {
  std::unordered_map<Node, SerialParallelDecomposition> node_to_sp;

  Node source = get_only(get_sources(g));
  node_to_sp.emplace(source, SerialParallelDecomposition(source)); // base-case

  for (Node const &node : get_topological_ordering(g)) {
    if (node == source) {
      continue;
    }
    std::unordered_set<SerialParallelDecomposition> sp_predecessors =
        transform(get_predecessors(g, node),
                  [&](Node const &p) { return node_to_sp.at(p); });

    SerialParallelDecomposition sp_decomp =
        normalize(serial_composition({parallel_composition(sp_predecessors),
                                      SerialParallelDecomposition(node)}));

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

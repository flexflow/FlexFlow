#include "utils/graph/sp_ization.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/multidigraph.h"
#include "utils/graph/serialparallel.h"
#include <cassert>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

bool is_2_terminal_sp_compliant(DiGraphView const &g) {
  return (is_acyclic(g) && has_single_source(g) && has_single_sink(g));
}

SerialParallelDecomposition
    barrier_sync_sp_ization_unchecked(DiGraphView const &g) {

  std::unordered_map<Node, int> node_to_sp_layer =
      get_longest_path_lengths_from_source_node(g);
  std::unordered_map<int, std::unordered_set<Node>> unordered_layer_to_node =
      invert_map(node_to_sp_layer);
  std::map<int, std::unordered_set<Node>> layer_to_node(
      unordered_layer_to_node.begin(), unordered_layer_to_node.end());

  Serial sp;
  for (auto const &[_, nodes] : layer_to_node) {
    Parallel layer{
        std::vector<std::variant<Serial, Node>>{nodes.begin(), nodes.end()}};
    sp.children.push_back(layer);
  }
  return normalize(sp);
}

SerialParallelDecomposition barrier_sync_sp_ization(DiGraphView const &g) {
  assert(is_2_terminal_sp_compliant(g));
  return barrier_sync_sp_ization_unchecked(g);
}

Serial cut_off_head(Serial s) {
  assert(s.children.size() > 0);
  return {std::vector<std::variant<Parallel, Node>>(s.children.begin() + 1,
                                                    s.children.end())};
}

SerialParallelDecomposition
    parallel_composition_with_coalescing(std::vector<Serial> sp_predecessors) {
  if (sp_predecessors.size() == 1) {
    return get_only(sp_predecessors);
  }
  std::map<std::variant<Parallel, Node>, std::vector<Serial>> coalescable;
  for (Serial predecessor : sp_predecessors) {
    if (predecessor.children.size() == 0) {
      continue;
    } else {
      coalescable[predecessor.children[0]].push_back(predecessor);
    }
  }

  std::vector<SerialParallelDecomposition> sp;
  for (auto const &item : coalescable) {
    std::variant<Parallel, Node> head = item.first;
    std::vector<Serial> sp_branches = item.second;
    std::vector<Serial> cut_off = transform(sp_branches, cut_off_head);
    auto p_comp = parallel_composition_with_coalescing(cut_off);
    sp.push_back(serial_composition({to_sp_decomp(head), p_comp}));
  }
  return parallel_composition(sp);
}

SerialParallelDecomposition
    dependency_invariant_sp_ization_unchecked_with_coalescing(
        DiGraphView const &g) {
  std::unordered_map<Node, Serial> node_to_sp;

  Node source = find_source_node(g);
  node_to_sp[source] = {{source}}; // base-case

  for (Node const &node : get_topological_ordering(g)) {
    if (node == source) {
      continue;
    }
    std::vector<Serial> sp_predecessors;
    for (Node const &p : get_predecessors(g, node)) {
      sp_predecessors.push_back(node_to_sp[p]);
    }
    SerialParallelDecomposition parallel_composed_predecessors =
        parallel_composition_with_coalescing(sp_predecessors);
    SerialParallelDecomposition sp_decomp =
        serial_composition({parallel_composed_predecessors, node});
    assert(std::holds_alternative<Serial>(sp_decomp));
    node_to_sp[node] = std::get<Serial>(sp_decomp);
  }

  Node sink = find_sink_node(g);
  return normalize(node_to_sp.at(sink));
}

SerialParallelDecomposition
    dependency_invariant_sp_ization_with_coalescing(DiGraphView const &g) {
  assert(is_2_terminal_sp_compliant(g));
  return dependency_invariant_sp_ization_unchecked_with_coalescing(g);
}

SerialParallelDecomposition
    dependency_invariant_sp_ization_unchecked(DiGraphView const &g) {
  std::unordered_map<Node, SerialParallelDecomposition> node_to_sp;

  Node source = find_source_node(g);
  node_to_sp[source] = source; // base-case

  for (Node const &node : get_topological_ordering(g)) {
    if (node == source) {
      continue;
    }
    std::unordered_set<SerialParallelDecomposition> unordered_sp_predecessors =
        transform(get_predecessors(g, node),
                  [&](Node const &p) { return node_to_sp[p]; });
    std::vector<SerialParallelDecomposition> sp_predecessors =
        as_vector(unordered_sp_predecessors);

    SerialParallelDecomposition sp_decomp =
        serial_composition({parallel_composition(sp_predecessors), node});
    node_to_sp[node] = sp_decomp;
  }

  Node sink = find_sink_node(g);
  return node_to_sp.at(sink);
}

SerialParallelDecomposition
    dependency_invariant_sp_ization(DiGraphView const &g) {
  assert(is_2_terminal_sp_compliant(g));
  return dependency_invariant_sp_ization_unchecked(g);
}

}; // namespace FlexFlow

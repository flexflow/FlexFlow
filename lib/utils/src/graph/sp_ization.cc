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

bool is_sp_compliant(DiGraphView const &g) {
  return (is_acyclic(g) && has_single_source(g) && has_single_sink(g));
}

SerialParallelDecomposition barrier_sync_sp_ization_unchecked(DiGraphView const &g) {

  std::unordered_map<Node, int> layer_map = get_longest_path_lengths(g);
  std::map<int, std::unordered_set<Node>> layers;
  for (Node const &node : get_nodes(g)) {
    int layer_num = layer_map[node];
    layers[layer_num].insert(node);
  }

  Serial sp;

  for (auto const &[_, nodes] : layers) {
    Parallel layer;
    for (Node const &node : nodes) {
      layer.children.push_back({node});
    }
    sp.children.push_back(layer);
  }
  return sp;
}

SerialParallelDecomposition barrier_sync_sp_ization(DiGraphView const &g) {
  assert(is_sp_compliant(g));
  return barrier_sync_sp_ization_unchecked(g);
}


// Original Recursive implementation, the current one is bottom up
// SerialParallelDecomposition helper_dependency_invariant_sp_ization(Node const &sink, DiGraphView const &g) {
//   std::vector<Nodes> predecessors = get_predecessors(sink, g);
//   if (predecessors.empty()) { //base case
//     return sink;
//   }
//   std::vector<SerialParallelDecomposition> sp_predecessors;
//   for (const auto& p : predecessors) {
//     sp_predecessors.push_back(helper_dependency_invariant_sp_ization(p, g));
//   }
//   SerialParallelDecomposition parallel_composed_predecessors = parallel_composed(sp_predecessors); 
//   return serial_composed({parallel_composed_predecessors, sink});
// }


// SerialParallelDecomposition naive_dependency_invariant_sp_ization_unchecked(DiGraphView const &g) {
//   Node sink = (*get_sinks(g).begin());
//   return helper_dependency_invariant_sp_ization(sink, g);
// }

SerialParallelDecomposition naive_dependency_invariant_sp_ization_unchecked(DiGraphView const &g) {
  std::vector<Node> topo_sorted_nodes = get_topological_ordering(g);
  std::unordered_map<Node, SerialParallelDecomposition> node_to_sp;

  Node source = find_source_node(g);
  node_to_sp[source] = source; //base-case

  for (const Node& node : topo_sorted_nodes) {
    if (node == source) {continue;}
    std::vector<SerialParallelDecomposition> sp_predecessors; //change to set
    for (const Node& p: get_predecessors(g, node)) {
      sp_predecessors.push_back(node_to_sp[p]);
    }

    SerialParallelDecomposition parallel_composed_predecessors = parallel_composition(sp_predecessors);
    SerialParallelDecomposition sp_decomp = serial_composition({parallel_composed_predecessors, node});
    node_to_sp[node] = sp_decomp;
  }

  Node sink = find_sink_node(g);
  return node_to_sp[sink];
}

SerialParallelDecomposition naive_dependency_invariant_sp_ization(DiGraphView const &g) {
  assert(is_sp_compliant(g));
  return naive_dependency_invariant_sp_ization_unchecked(g);
}

}; // namespace FlexFlow

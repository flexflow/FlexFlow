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

SerialParallelDecomposition
    barrier_sync_sp_ization_unchecked(DiGraphView const &g) {

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

Serial cut_off_head(Serial s) {
  assert(s.children.size()>0);
  return {std::vector<std::variant<Parallel, Node>>(s.children.begin()+1, s.children.end())};
}

SerialParallelDecomposition parallel_composition_with_coalescing(std::vector<Serial> sp_predecessors) {
  if (sp_predecessors.size()==1) {return get_only(sp_predecessors);}
  std::unordered_map<Node, std::vector<Serial>> coalescable;
  std::vector<Serial> non_coalescable;
  for (Serial predecessor : sp_predecessors) {
    if (predecessor.children.size() == 0) {continue;}
    else if (std::holds_alternative<Node>(predecessor.children[0])) {
      Node n = std::get<Node>(predecessor.children[0]);
      coalescable[n].push_back(predecessor); //TODO: apply the cut
    }
    else if (std::holds_alternative<Parallel>(predecessor.children[0])) {
      non_coalescable.push_back(predecessor);
    }
  }
  std::vector<SerialParallelDecomposition> sp;
  for (const auto& item : coalescable) {
    Node head = item.first;
    std::vector<Serial> sp_branches = item.second;
    std::vector<Serial> cut_off = transform(sp_branches, cut_off_head);
    auto p_comp = parallel_composition(transform(cut_off, [](Serial s) -> SerialParallelDecomposition {return s;}));
    sp.push_back(serial_composition({head, p_comp}));
  }
  for (Serial const &nc : non_coalescable) {
    sp.push_back(nc);
  }
  return parallel_composition(sp);
}

SerialParallelDecomposition dependency_invariant_sp_ization_unchecked_with_coalescing(DiGraphView const &g) {
  std::vector<Node> topo_sorted_nodes = get_topological_ordering(g);
  std::unordered_map<Node, Serial> node_to_sp;
  
  Node source = find_source_node(g);
  node_to_sp[source] = {{source}}; // base-case

  for (Node const &node : topo_sorted_nodes) {
    if (node == source) {
      continue;
    }
    std::vector<Serial> sp_predecessors;
    for (Node const &p : get_predecessors(g, node)) {
      sp_predecessors.push_back(node_to_sp[p]);
    }
    SerialParallelDecomposition parallel_composed_predecessors = parallel_composition_with_coalescing(sp_predecessors);
    SerialParallelDecomposition sp_decomp = serial_composition({parallel_composed_predecessors, node});
    assert(std::holds_alternative<Serial>(sp_decomp));
    node_to_sp[node] = std::get<Serial>(sp_decomp);
  }

  Node sink = find_sink_node(g);
  return normalize(node_to_sp[sink]);
}

SerialParallelDecomposition
    dependency_invariant_sp_ization_with_coalescing(DiGraphView const &g) {
  assert(is_sp_compliant(g));
  return dependency_invariant_sp_ization_unchecked_with_coalescing(g);
}

SerialParallelDecomposition dependency_invariant_sp_ization_unchecked(DiGraphView const &g) {
  std::vector<Node> topo_sorted_nodes = get_topological_ordering(g);
  std::unordered_map<Node, SerialParallelDecomposition> node_to_sp;
  
  Node source = find_source_node(g);
  node_to_sp[source] = source; // base-case

  for (Node const &node : topo_sorted_nodes) {
    if (node == source) {
      continue;
    }
    std::vector<SerialParallelDecomposition> sp_predecessors;
    for (Node const &p : get_predecessors(g, node)) {
      sp_predecessors.push_back(node_to_sp[p]);
    }
    SerialParallelDecomposition parallel_composed_predecessors = parallel_composition(sp_predecessors);
    SerialParallelDecomposition sp_decomp = serial_composition({parallel_composed_predecessors, node});
    node_to_sp[node] = sp_decomp;
  }

  Node sink = find_sink_node(g);
  return node_to_sp[sink];
}

SerialParallelDecomposition
    dependency_invariant_sp_ization(DiGraphView const &g) {
  assert(is_sp_compliant(g));
  return dependency_invariant_sp_ization_unchecked(g);
}

}; // namespace FlexFlow

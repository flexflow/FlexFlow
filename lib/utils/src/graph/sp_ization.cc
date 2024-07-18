#include "utils/graph/sp_ization.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/multidigraph.h"
#include "utils/graph/serialparallel.h"
#include <cassert>
#include <queue>

namespace FlexFlow {

bool is_2_terminal_sp_compliant(DiGraphView const &g) {
  return (is_acyclic(g) && has_single_source(g) && has_single_sink(g));
}

std::vector<std::unordered_set<Node>> naive_layer_split(DiGraphView const &g) {
  std::unordered_map<Node, int> node_to_sp_layer =
      get_longest_path_lengths_from_source_node(g);
  std::unordered_map<int, std::unordered_set<Node>> unordered_layer_to_node =
      invert_map(node_to_sp_layer);
  std::vector<std::unordered_set<Node>> layer_to_nodes;
  for (int layer_num : sorted(keys(unordered_layer_to_node))) {
    layer_to_nodes.push_back(unordered_layer_to_node[layer_num]);
  }
  return layer_to_nodes;
}

SerialParallelDecomposition
    naive_layer_merge(std::vector<std::unordered_set<Node>> layer_to_node) {
  Serial sp;
  for (auto const nodes : layer_to_node) {
    Parallel layer{
        std::vector<std::variant<Serial, Node>>{nodes.begin(), nodes.end()}};
    sp.children.push_back(layer);
  }
  return normalize(sp);
}

std::unordered_set<Node> get_heads(DiGraphView const &g,
                                   std::vector<DiGraphView> metanodes,
                                   std::unordered_set<Node> explored) {
  std::unordered_set<Node> previous_layer_sinks =
      set_union(transform(metanodes, get_sinks));
  std::unordered_set<Node> candidate_heads = set_union(values(get_successors(
      flipped(g), previous_layer_sinks))); // TODO: remove flipped, accounts for
                                           // diedge bug currently
  return filter(candidate_heads, [&](Node const &n) {
    return all_of(get_predecessors(flipped(g), n),
                  [&](Node const &p) { return contains(explored, p); });
  }); // TODO: remove flipped, accounts for diedge bug currently
}

std::unordered_set<std::vector<Node>> get_non_overlapping_topological_orderings(
    DiGraphView const &g, std::unordered_set<Node> const &heads) {
  std::unordered_set<std::vector<Node>> topo_orderings =
      transform(heads, [&](Node const &head) {
        return get_topological_ordering_from_starting_node(flipped(g),
                                                           {{head}});
      }); // TODO: remove flipped, accounts for diedge bug currently
  std::unordered_set<Node> all_nodes = set_union(
      without_order(transform(topo_orderings, [&](std::vector<Node> const &v) {
        return without_order(v);
      })));
  std::unordered_set non_visitable_nodes =
      filter(all_nodes, [&](Node const &n) {
        return sum(transform(topo_orderings, [&](auto const &ordering) {
                 return contains(ordering, n);
               })) != 1;
      });
  std::unordered_set<std::vector<Node>> non_overlapping_topo_orderings;
  for (auto const &topo_ordering : topo_orderings) {
    non_overlapping_topo_orderings.insert(
        filter(topo_ordering, [&](Node const &n) {
          return !contains(non_visitable_nodes, n);
        }));
  }
  return non_overlapping_topo_orderings;
}

std::vector<DiGraphView> get_metanodes(
    DiGraphView const &g,
    std::unordered_set<std::vector<Node>> const &non_overlapping_topo_orderings,
    float layer_cost,
    std::unordered_map<Node, float> const &cost_map) {
  std::vector<DiGraphView> metanodes;
  for (auto const topo_ordering : non_overlapping_topo_orderings) {
    float s = 0;
    std::vector<Node> explored_nodes;
    for (Node const &node : topo_ordering) {
      if (cost_map.at(node) + s > layer_cost) {
        break;
      } else {
        explored_nodes.push_back(node);
        s += cost_map.at(node);
      }
    }
    metanodes.push_back(get_subgraph(g, without_order(explored_nodes)));
  }
  return metanodes;
}

std::vector<std::vector<DiGraphView>>
    cost_aware_layer_split(DiGraphView const &g,
                           std::unordered_map<Node, float> const &cost_map) {
  std::vector<std::vector<DiGraphView>> layers;
  Node source = get_only(get_sinks(g));
  std::unordered_set<Node> explored = {{source}};
  layers.push_back({{get_subgraph(g, {{source}})}});
  for (int i = 1; get_nodes(g) != explored; i++) {
    std::unordered_set<Node> heads = get_heads(g, layers.at(i - 1), explored);
    float layer_cost = maximum(
        transform(heads, [&](Node const &n) { return cost_map.at(n); }));
    std::unordered_set<std::vector<Node>> non_overlapping_topo_orderings =
        get_non_overlapping_topological_orderings(g, heads);
    std::vector<DiGraphView> metanodes =
        get_metanodes(g, non_overlapping_topo_orderings, layer_cost, cost_map);
    layers.push_back(metanodes);
    std::unordered_set<Node> newly_explored = set_union(transform(
        metanodes, [](DiGraphView const &g) { return get_nodes(g); }));
    explored = set_union(explored, newly_explored);
  }
  return layers;
}

SerialParallelDecomposition cost_aware_barrier_sync_sp_ization_unchecked(
    DiGraphView const &g, std::unordered_map<Node, float> const &cost_map) {
  if (get_nodes(g).size() == 1) {
    return get_only(get_nodes(g));
  }
  std::vector<std::vector<DiGraphView>> layer_split =
      cost_aware_layer_split(g, cost_map);
  std::vector<std::vector<SerialParallelDecomposition>> sp_ized_layers =
      transform(layer_split, [&](std::vector<DiGraphView> const &layer) {
        return transform(layer, [&](DiGraphView const &g) {
          return cost_aware_barrier_sync_sp_ization_unchecked(g, cost_map);
        });
      });
  return normalize(serial_composition(transform(
      sp_ized_layers, [](std::vector<SerialParallelDecomposition> const &sp) {
        return parallel_composition(sp);
      })));
}

SerialParallelDecomposition cost_aware_barrier_sync_sp_ization(
    DiGraphView const &g, std::unordered_map<Node, float> const &cost_map) {
  assert(is_2_terminal_sp_compliant(g));
  return cost_aware_barrier_sync_sp_ization_unchecked(g, cost_map);
}

SerialParallelDecomposition
    barrier_sync_sp_ization_unchecked(DiGraphView const &g) {

  std::vector<std::unordered_set<Node>> layer_split = naive_layer_split(g);
  return naive_layer_merge(layer_split);
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

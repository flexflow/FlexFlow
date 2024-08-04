#include "utils/containers.h"
#include "utils/containers/all_of.h"
#include "utils/containers/as_unordered_set.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/invert_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/sorted.h"
#include "utils/containers/values.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/algorithms/is_2_terminal_dag.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/parallel_composition.h"
#include "utils/graph/serial_parallel/serial_composition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_metrics.h"
#include "utils/graph/serial_parallel/serial_parallel_normalize.h"
#include "utils/hash/vector.h"
#include <unordered_set>
#include <vector>
#include "utils/graph/digraph/algorithms/is_acyclic.h"

namespace FlexFlow {

std::vector<std::unordered_set<Node>> naive_layer_split(DiGraphView const &g) {
  std::unordered_map<Node, int> node_to_sp_layer =
      get_longest_path_lengths_from_root(g);
  std::unordered_map<int, std::unordered_set<Node>> unordered_layer_to_node =
      invert_map(node_to_sp_layer);
  std::vector<std::unordered_set<Node>> layer_to_nodes;
  for (int layer_num : sorted(keys(unordered_layer_to_node))) {
    layer_to_nodes.push_back(unordered_layer_to_node.at(layer_num));
  }
  return layer_to_nodes;
}

static SerialParallelDecomposition
    naive_layer_merge(std::vector<std::unordered_set<Node>> layer_to_node) {
  SerialSplit sp({});
  for (auto const &nodes : layer_to_node) {
    ParallelSplit layer({});
    for (Node const &node : nodes) {
      layer.children.insert(node);
    }
    sp.children.push_back(layer);
  }
  return normalize(SerialParallelDecomposition(sp));
}

SerialParallelDecomposition
    barrier_sync_sp_ization_unchecked(DiGraphView const &g) {

  std::vector<std::unordered_set<Node>> layer_split = naive_layer_split(g);
  return naive_layer_merge(layer_split);
}

SerialParallelDecomposition barrier_sync_sp_ization(DiGraphView const &g) {
  assert(is_acyclic(g));
  return barrier_sync_sp_ization_unchecked(g);
}

static std::unordered_set<Node> get_heads(DiGraphView const &g,
                                          std::vector<DiGraphView> metanodes,
                                          std::unordered_set<Node> explored) {
  std::unordered_set<Node> previous_layer_nodes = set_union(
      transform(metanodes, [&](DiGraphView const &g) { return get_nodes(g); }));
  std::unordered_set<Node> candidate_heads =
      set_union(values(get_successors(g, previous_layer_nodes)));
  return filter(candidate_heads, [&](Node const &n) {
    return (!contains(explored, n) &&
            all_of(get_predecessors(g, n),
                   [&](Node const &p) { return contains(explored, p); }));
  });
}

static std::unordered_set<std::vector<Node>>
    get_non_overlapping_topological_orderings(
        DiGraphView const &g, std::unordered_set<Node> const &heads) {
  std::unordered_set<std::vector<Node>> topo_orderings =
      transform(heads, [&](Node const &head) {
        return get_topological_ordering_from_starting_node(g, head);
      });
  std::unordered_set<Node> all_nodes = set_union(as_unordered_set(
      transform(topo_orderings, [&](std::vector<Node> const &v) {
        return as_unordered_set(v);
      })));

  std::unordered_set<Node> non_visitable_nodes =
      filter(all_nodes, [&](Node const &n) {
        std::vector<int> contains_vec = transform(
            as_vector(topo_orderings), [&](auto const &ordering) -> int {
              return contains(ordering, n) ? 1 : 0;
            });
        return sum(contains_vec) != 1;
      });
  std::unordered_set<std::vector<Node>> non_overlapping_topo_orderings;
  for (std::vector<Node> const &topo_ordering : topo_orderings) {
    // std::vector<Node> filtered_topo_ordering = filter(topo_ordering,
    // [&](Node
    // const &n) {return !contains(non_visitable_nodes, n);});
    std::vector<Node> filtered_topo_ordering;
    for (Node const &n : topo_ordering) {
      if (!contains(non_visitable_nodes, n)) {
        filtered_topo_ordering.push_back(n);
      }
    }
    non_overlapping_topo_orderings.insert(filtered_topo_ordering);
  }
  return non_overlapping_topo_orderings;
}

static std::vector<DiGraphView> get_metanodes_naive(
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
    metanodes.push_back(get_subgraph(g, as_unordered_set(explored_nodes)));
  }
  return metanodes;
}

static std::vector<DiGraphView> get_metanodes(
    DiGraphView const &g,
    std::unordered_set<std::vector<Node>> const &non_overlapping_topo_orderings,
    float layer_cost,
    std::unordered_map<Node, float> const &cost_map) {
  std::vector<DiGraphView> metanodes;
  for (auto const topo_ordering : non_overlapping_topo_orderings) {
    std::vector<Node> explored_nodes;
    for (Node const &node : topo_ordering) {
      explored_nodes.push_back(node);
      DiGraphView sg = get_subgraph(g, as_unordered_set(explored_nodes));
      if (critical_path_cost(barrier_sync_sp_ization(sg), cost_map) >
          layer_cost * 1.01) {
        explored_nodes.pop_back();
        break;
      }
    }
    metanodes.push_back(get_subgraph(g, as_unordered_set(explored_nodes)));
  }
  return metanodes;
}

static std::vector<std::vector<DiGraphView>>
    cost_aware_layer_split(DiGraphView const &g,
                           std::unordered_map<Node, float> const &cost_map) {
  std::vector<std::vector<DiGraphView>> layers;
  Node source = get_only(get_sources(g));
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
    return SerialParallelDecomposition(get_only(get_nodes(g)));
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
  // assert(is_2_terminal_dag(g));
  return cost_aware_barrier_sync_sp_ization_unchecked(g, cost_map);
}

} // namespace FlexFlow

#include "utils/graph/serial_parallel/sp_ization/work_preserving_sp_ization.h"
#include "utils/containers.h"
#include "utils/containers/all_of.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/invert_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/sorted.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering_from_starting_node.h"
#include "utils/graph/digraph/algorithms/is_2_terminal_dag.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/normalize_sp_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_metrics.h"
#include "utils/hash/unordered_set.h"
#include "utils/hash/vector.h"
#include <unordered_set>
#include <vector>

namespace FlexFlow {

std::vector<std::unordered_set<Node>>
    stratum_split_assuming_unit_cost(DiGraphView const &g) {
  std::unordered_map<Node, int> node_to_stratum =
      get_longest_path_lengths_from_root(g);
  std::vector<std::unordered_set<Node>> result(
      maximum(values(node_to_stratum)));
  for (auto const &[node, depth] : node_to_stratum) {
    result[depth - 1].insert(node);
  }
  return result;
}

static SerialParallelDecomposition
    naive_stratum_merge(std::vector<std::unordered_set<Node>> stratum_split) {
  std::vector<SerialParallelDecomposition> strata = transform(
      stratum_split, [](std::unordered_set<Node> const &stratum_nodes) {
        return SerialParallelDecomposition(ParallelSplit{stratum_nodes});
      });
  return normalize_sp_decomposition(serial_composition(strata));
}

SerialParallelDecomposition
    stratum_sync_sp_ization_unchecked(DiGraphView const &g) {

  std::vector<std::unordered_set<Node>> stratum_split =
      stratum_split_assuming_unit_cost(g);
  return naive_stratum_merge(stratum_split);
}

SerialParallelDecomposition stratum_sync_sp_ization(DiGraphView const &g) {
  assert(is_acyclic(g));
  return stratum_sync_sp_ization_unchecked(g);
}

static std::unordered_set<Node> get_heads(
    DiGraphView const &g,
    std::unordered_set<std::unordered_set<Node>> previous_stratum_metanodes,
    std::unordered_set<Node> explored) {
  std::unordered_set<Node> previous_stratum_nodes =
      set_union(previous_stratum_metanodes);
  std::unordered_set<Node> candidate_heads =
      set_union(values(get_successors(g, previous_stratum_nodes)));

  auto is_valid_head = [&](Node const &n) {
    return (!contains(explored, n) &&
            all_of(get_predecessors(g, n),
                   [&](Node const &p) { return contains(explored, p); }));
  };

  return filter(candidate_heads, is_valid_head);
}

// returns a set of filtered topological orderings starting from `heads` such
// that all nodes present in multiple orderings are not included
static std::unordered_set<std::vector<Node>>
    get_non_overlapping_topological_orderings(
        DiGraphView const &g, std::unordered_set<Node> const &heads) {

  std::unordered_set<std::vector<Node>> topo_orderings =
      transform(heads, [&](Node const &head) {
        return get_topological_ordering_from_starting_node(g, head);
      });

  std::unordered_map<Node, int> node_frequency;
  for (std::vector<Node> const &ordering : topo_orderings) {
    for (Node const &node : ordering) {
      node_frequency[node]++;
    }
  }

  std::unordered_set<Node> visitable_nodes =
      filter(keys(node_frequency),
             [&](Node const &n) { return node_frequency.at(n) == 1; });

  std::unordered_set<std::vector<Node>> non_overlapping_topo_orderings =
      transform(topo_orderings, [&](std::vector<Node> const &ordering) {
        return filter(ordering, [&](Node const &n) {
          return contains(visitable_nodes, n);
        });
      });

  return non_overlapping_topo_orderings;
}

static std::unordered_set<std::unordered_set<Node>>
    get_metanodes(DiGraphView const &g,
                  std::unordered_set<std::vector<Node>> const &topo_orderings,
                  float stratum_cost,
                  std::unordered_map<Node, float> const &cost_map) {

  auto get_metanode = [&](std::vector<Node> const &topo_ordering) {
    std::unordered_set<Node> explored_nodes;
    for (Node const &node : topo_ordering) {
      float metanode_cost =
          critical_path_cost(stratum_sync_sp_ization(get_subgraph(
                                 g, set_union(explored_nodes, {node}))),
                             cost_map);
      if (metanode_cost > stratum_cost * 1.01) {
        break;
      }
      explored_nodes.insert(node);
    }
    return explored_nodes;
  };

  return transform(topo_orderings, get_metanode);
}

static std::vector<std::unordered_set<std::unordered_set<Node>>>
    cost_aware_stratum_split(DiGraphView const &g,
                             std::unordered_map<Node, float> const &cost_map) {
  std::vector<std::unordered_set<std::unordered_set<Node>>> strata;
  Node source = get_only(get_sources(g));
  std::unordered_set<Node> explored = {source};
  strata.push_back({{source}});
  while (get_nodes(g) != explored) {

    std::unordered_set<Node> heads = get_heads(g, strata.back(), explored);
    std::unordered_set<std::vector<Node>> non_overlapping_topo_orderings =
        get_non_overlapping_topological_orderings(g, heads);
    float stratum_cost = maximum(
        transform(heads, [&](Node const &n) { return cost_map.at(n); }));
    std::unordered_set<std::unordered_set<Node>> metanodes = get_metanodes(
        g, non_overlapping_topo_orderings, stratum_cost, cost_map);
    strata.push_back(metanodes);

    explored = set_union(explored, set_union(metanodes));
  }
  return strata;
}

SerialParallelDecomposition cost_aware_stratum_sync_sp_ization_unchecked(
    DiGraphView const &g, std::unordered_map<Node, float> const &cost_map) {
  if (get_nodes(g).size() == 1) {
    return SerialParallelDecomposition(get_only(get_nodes(g)));
  }
  std::vector<std::unordered_set<std::unordered_set<Node>>> stratum_split =
      cost_aware_stratum_split(g, cost_map);
  std::vector<std::unordered_set<SerialParallelDecomposition>> sp_ized_strata;
  for (auto const &stratum : stratum_split) {
    auto sp_ized_stratum =
        transform(stratum, [&](std::unordered_set<Node> const &nodes) {
          return cost_aware_stratum_sync_sp_ization_unchecked(
              get_subgraph(g, nodes), cost_map);
        });
    sp_ized_strata.push_back(sp_ized_stratum);
  }

  return normalize_sp_decomposition(
      serial_composition(transform(sp_ized_strata, parallel_composition)));
}

SerialParallelDecomposition cost_aware_stratum_sync_sp_ization(
    DiGraphView const &g, std::unordered_map<Node, float> const &cost_map) {
  assert(is_acyclic(g));
  return cost_aware_stratum_sync_sp_ization_unchecked(g, cost_map);
}

} // namespace FlexFlow

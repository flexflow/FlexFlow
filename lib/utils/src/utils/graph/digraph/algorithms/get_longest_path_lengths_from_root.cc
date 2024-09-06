#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/containers.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include <unordered_map>

// change to have unweigthed call the weighted version

namespace FlexFlow {

std::unordered_map<Node, float> get_weighted_longest_path_lengths_from_root(
    DiGraphView const &g, std::unordered_map<Node, float> const &node_costs) {

  assert(is_acyclic(g));
  assert(get_sources(g).size() == 1);

  std::vector<Node> topo_order = get_topological_ordering(g);
  std::unordered_map<Node, float> longest_path_lengths;

  for (Node const &n : topo_order) {
    std::unordered_set<float> predecessor_path_lengths =
        transform(get_predecessors(g, n), [&](Node const &pred) {
          return longest_path_lengths.at(pred);
        });
    longest_path_lengths[n] =
        (predecessor_path_lengths.size() == 0)
            ? node_costs.at(n)
            : maximum(predecessor_path_lengths) + node_costs.at(n);
  }
  return longest_path_lengths;
}

std::unordered_map<Node, int>
    get_longest_path_lengths_from_root(DiGraphView const &g) {

  assert(is_acyclic(g));
  assert(get_sources(g).size() == 1);

  std::vector<Node> topo_order = get_topological_ordering(g);
  std::unordered_map<Node, int> longest_path_lengths;

  for (Node const &n : topo_order) {
    std::unordered_set<int> predecessor_path_lengths =
        transform(get_predecessors(g, n), [&](Node const &pred) {
          return longest_path_lengths.at(pred);
        });
    longest_path_lengths[n] = (predecessor_path_lengths.size() == 0)
                                  ? 1
                                  : maximum(predecessor_path_lengths) + 1;
  }

  return longest_path_lengths;
}

} // namespace FlexFlow

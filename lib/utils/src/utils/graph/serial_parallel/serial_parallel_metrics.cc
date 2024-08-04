#include "utils/graph/serial_parallel/serial_parallel_metrics.h"
#include "utils/containers.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <unordered_map>

// TODO: change the getting children split into 2 into a single case

namespace FlexFlow {

std::unordered_map<Node, size_t>
    node_frequency_counter(SerialParallelDecomposition const &sp) {
  std::unordered_map<Node, size_t> counter;

  if (sp.has<Node>()) {
    Node node = sp.get<Node>();
    counter[node]++;
  } else if (sp.has<SerialSplit>()) {
    for (std::variant<ParallelSplit, Node> const &child :
         sp.get<SerialSplit>().children) {
      std::unordered_map<Node, size_t> child_counter =
          node_frequency_counter(widen<SerialParallelDecomposition>(child));
      for (auto const &[node, count] : child_counter) {
        counter[node] += count;
      }
    }
  } else {
    assert(sp.has<ParallelSplit>());
    for (std::variant<SerialSplit, Node> const &child :
         sp.get<ParallelSplit>().children) {
      std::unordered_map<Node, size_t> child_counter =
          node_frequency_counter(widen<SerialParallelDecomposition>(child));
      for (auto const &[node, count] : child_counter) {
        counter[node] += count;
      }
    }
  }

  return counter;
}

// duplicate nodes are counted multiple times
size_t num_nodes(SerialParallelDecomposition const &sp) {
  return sum(values(node_frequency_counter(sp)));
}

float work_cost(SerialParallelDecomposition const &sp,
                std::unordered_map<Node, float> cost_map) {
  auto cost_per_node_group = [&](std::pair<Node, float> const &pair) {
    return pair.second * cost_map.at(pair.first);
  };
  std::unordered_map<Node, size_t> counter = node_frequency_counter(sp);
  std::vector<std::pair<Node, size_t>> pairs(counter.cbegin(), counter.cend());
  return sum(transform(pairs, cost_per_node_group));
}

float work_cost(DiGraphView const &g,
                std::unordered_map<Node, float> const &cost_map) {
  return sum(transform(as_vector(get_nodes(g)),
                       [&](Node const &node) { return cost_map.at(node); }));
}

float critical_path_cost(SerialParallelDecomposition const &sp,
                         std::unordered_map<Node, float> const &cost_map) {
  if (sp.has<Node>()) {
    return cost_map.at(sp.get<Node>());
  } else if (sp.has<SerialSplit>()) {
    return sum(transform(sp.get<SerialSplit>().children,
                         [&](std::variant<ParallelSplit, Node> const &child) {
                           return critical_path_cost(
                               widen<SerialParallelDecomposition>(child),
                               cost_map);
                         }));
  } else {
    assert(sp.has<ParallelSplit>());
    return maximum(transform(sp.get<ParallelSplit>().children,
                             [&](std::variant<SerialSplit, Node> const &child) {
                               return critical_path_cost(
                                   widen<SerialParallelDecomposition>(child),
                                   cost_map);
                             }));
  }
}

float critical_path_cost(DiGraphView const &g,
                         std::unordered_map<Node, float> const &cost_map) {
  return maximum(
      values(get_weighted_longest_path_lengths_from_root(g, cost_map)));
}

float average_parallelism_degree(
    SerialParallelDecomposition const &sp,
    std::unordered_map<Node, float> const &cost_map) {
  NOT_IMPLEMENTED();
}

// TODO: delete the graph functions as instead convert SP to graph and then pass
// it to the other functions

float max_parallelism_degree(SerialParallelDecomposition const &sp,
                             std::unordered_map<Node, float> const &cost_map) {
  NOT_IMPLEMENTED();
}

float relative_work_increase(DiGraphView const &g,
                             SerialParallelDecomposition const &sp,
                             std::unordered_map<Node, float> const &cost_map) {
  return work_cost(sp, cost_map) / work_cost(g, cost_map);
}

float relative_critical_path_cost_increase(
    DiGraphView const &g,
    SerialParallelDecomposition const &sp,
    std::unordered_map<Node, float> const &cost_map) {
  return critical_path_cost(sp, cost_map) / critical_path_cost(g, cost_map);
}

} // namespace FlexFlow

#include "utils/graph/digraph/algorithms/get_dominators_map.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/node/algorithms.h"
#include "utils/hash/unordered_set.h"
#include <queue>

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<Node>>
    get_dominators_map(DiGraphView const &g) {
  std::unordered_set<Node> sources = get_sources(g);

  std::queue<Node> queue;

  for (Node src : get_sources(g)) {
    queue.push(src);
  }

  std::unordered_map<Node, std::unordered_set<Node>> result =
      generate_map(get_nodes(g), [&](Node const &) { return get_nodes(g); });
  while (!queue.empty()) {
    Node n = queue.front();
    queue.pop();

    std::unordered_set<Node> old_result_entry = result.at(n);

    result.at(n) =
        intersection(transform(get_predecessors(g, n), [&](Node const &n) {
          return result.at(n);
        })).value_or(std::unordered_set<Node>{});
    result.at(n).insert(n);

    if (result.at(n) != old_result_entry) {
      for (Node const &succ : get_successors(g, n)) {
        queue.push(succ);
      }
    }
  }

  return result;
}

} // namespace FlexFlow

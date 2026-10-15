#include "utils/graph/digraph/algorithms/get_dominators_map.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/set_intersection.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/node/algorithms.h"
#include "utils/hash/set.h"
#include <optional>
#include <queue>

namespace FlexFlow {

std::map<Node, std::set<Node>> get_dominators_map(DiGraphView const &g) {
  std::set<Node> initial_nodes = get_initial_nodes(g);

  std::queue<Node> queue;

  for (Node src : get_initial_nodes(g)) {
    queue.push(src);
  }

  std::map<Node, std::set<Node>> result =
      generate_map(get_nodes(g), [&](Node const &) { return get_nodes(g); });
  while (!queue.empty()) {
    Node n = queue.front();
    queue.pop();

    std::set<Node> old_result_entry = result.at(n);

    result.at(n) =
        set_intersection(transform(get_predecessors(g, n), [&](Node const &n) {
          return result.at(n);
        })).value_or(std::set<Node>{});
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

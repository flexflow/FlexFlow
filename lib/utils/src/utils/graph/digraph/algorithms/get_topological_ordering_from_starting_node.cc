#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/traversal.h"

namespace FlexFlow {

static std::vector<Node> get_unchecked_topological_ordering_from_starting_node(
    DiGraphView const &g, Node const &starting_node) {

  auto get_descendants = [&](DiGraphView const &g, Node const &starting_node) {
    std::unordered_set<Node> descendants;
    std::stack<Node> to_visit;
    to_visit.push(starting_node);
    while (!to_visit.empty()) {
      Node current = to_visit.top();
      to_visit.pop();
      descendants.insert(current);
      for (auto const &s :
           filter(get_successors(g, current),
                  [&](Node const &n) { return !contains(descendants, n); })) {
        to_visit.push(s);
      }
    }
    return descendants;
  };
  std::unordered_set<Node> descendants = get_descendants(g, starting_node);
  return get_topological_ordering(get_subgraph(g, descendants));
}

std::vector<Node>
    get_topological_ordering_from_starting_node(DiGraphView const &g,
                                                Node const &starting_node) {
  assert(is_acyclic(g));
  return get_unchecked_topological_ordering_from_starting_node(g,
                                                               starting_node);
}

} // namespace FlexFlow

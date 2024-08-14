#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {
std::unordered_set<Node> get_descendants(DiGraphView const &g,
                                         Node const &starting_node) {
  assert(is_acyclic(g));
  std::unordered_set<Node> descendants;
  std::stack<Node> to_visit;
  for (Node const &successor : get_successors(g, starting_node)) {
    to_visit.push(successor);
  }
  while (!to_visit.empty()) {
    Node current = to_visit.top();
    to_visit.pop();
    descendants.insert(current);
    for (auto const &s : filter(get_successors(g, current), [&](Node const &n) {
           return !contains(descendants, n);
         })) {
      to_visit.push(s);
    }
  }
  return descendants;
};

} // namespace FlexFlow

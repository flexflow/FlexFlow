#include "utils/graph/digraph/algorithms/get_dominators.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/values.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/hash/unordered_set.h"

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<Node>>
    get_dominators(DiGraphView const &g) {
  std::vector<Node> topo = get_topological_ordering(g);
  std::unordered_map<Node, std::unordered_set<Node>> result;

  for (Node const &n : topo) {
    result[n] =
        intersection(transform(get_predecessors(g, n), [&](Node const &n) {
          return result.at(n);
        })).value_or(std::unordered_set<Node>{});
    ;
    result[n].insert(n);
  }

  return result;
}

std::unordered_set<Node> get_dominators(DiGraphView const &g, Node const &n) {
  return get_dominators(g).at(n);
}

std::unordered_set<Node> get_dominators(DiGraphView const &g,
                                        std::unordered_set<Node> const &n) {
  if (n.empty()) {
    throw mk_runtime_error("Cannot find dominators of no nodes");
  }
  std::optional<std::unordered_set<Node>> result =
      intersection(values(restrict_keys(get_dominators(g), n)));
  assert(result.has_value());

  return result.value();
}

} // namespace FlexFlow

#include "utils/graph/digraph/algorithms/get_dominators.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_dominators_map.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/node/algorithms.h"
#include "utils/hash/unordered_set.h"
#include <queue>

namespace FlexFlow {

std::unordered_set<Node> get_dominators(DiGraphView const &g, Node const &n) {
  return get_dominators_map(g).at(n);
}

std::unordered_set<Node> get_dominators(DiGraphView const &g,
                                        std::unordered_set<Node> const &n) {
  if (n.empty()) {
    throw mk_runtime_error("Cannot find dominators of no nodes");
  }
  std::optional<std::unordered_set<Node>> result =
      intersection(values(restrict_keys(get_dominators_map(g), n)));
  assert(result.has_value());

  return result.value();
}

} // namespace FlexFlow

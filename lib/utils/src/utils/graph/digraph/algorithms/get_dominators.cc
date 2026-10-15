#include "utils/graph/digraph/algorithms/get_dominators.h"
#include "utils/containers/restrict_keys_strict.h"
#include "utils/containers/set_intersection.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/get_dominators_map.h"
#include "utils/hash/set.h"
#include "utils/optional.h"
#include <queue>

namespace FlexFlow {

std::set<Node> get_dominators(DiGraphView const &g, Node const &n) {
  return get_dominators_map(g).at(n);
}

std::set<Node> get_dominators(DiGraphView const &g, std::set<Node> const &n) {
  ASSERT(n.size() > 0, "Cannot find dominators of no nodes");

  std::optional<std::set<Node>> result =
      set_intersection(values(restrict_keys_strict(get_dominators_map(g), n)));

  return assert_unwrap(result);
}

} // namespace FlexFlow

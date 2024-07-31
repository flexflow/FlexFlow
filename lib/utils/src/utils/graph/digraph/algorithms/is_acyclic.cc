#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/traversal.h"

namespace FlexFlow {

std::optional<bool> is_acyclic(DiGraphView const &g) {
  if (num_nodes(g) == 0) {
    return std::nullopt;
  }
  std::unordered_set<Node> sources = get_sources(g);
  if (sources.size() == 0) {
    return false;
  }
  auto dfs_view = unchecked_dfs(g, sources);
  std::unordered_set<Node> seen;
  for (unchecked_dfs_iterator it = dfs_view.begin(); it != dfs_view.end();
       it++) {
    if (contains(seen, *it)) {
      return false;
    } else {
      seen.insert(*it);
    }
  }
  if (seen != get_nodes(g)) {
    return false;
  }
  return true;
}

} // namespace FlexFlow

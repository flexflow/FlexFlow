#include "utils/graph/digraph/algorithms/get_imm_post_dominator.h"
#include "utils/graph/digraph/algorithms/get_node_with_greatest_topo_rank.h"
#include "utils/graph/digraph/algorithms/get_post_dominators.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/values.h"
#include "utils/containers/intersection.h"
#include "utils/optional.h"
#include "utils/graph/digraph/algorithms/get_imm_post_dominators.h"

namespace FlexFlow {

std::optional<Node> get_imm_post_dominator(DiGraphView const &g, Node const &n) {
  return get_imm_post_dominators(g).at(n);
}

std::optional<Node>
    get_imm_post_dominator(DiGraphView const &g,
                           std::unordered_set<Node> const &nodes) {

  if (nodes.empty()) {
    throw mk_runtime_error("Cannot get imm_post_dominator of no nodes");
  }
  std::unordered_set<Node> commonDoms = assert_unwrap(
      intersection(values(restrict_keys(get_post_dominators(g), nodes))));

  if (!commonDoms.empty()) {
    return get_node_with_greatest_topo_rank(commonDoms, g);
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow

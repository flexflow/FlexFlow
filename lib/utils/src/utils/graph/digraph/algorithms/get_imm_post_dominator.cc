#include "utils/graph/digraph/algorithms/get_imm_post_dominator.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_first.h"
#include "utils/containers/get_only.h"
#include "utils/containers/intersection.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/apply_contraction.h"
#include "utils/graph/digraph/algorithms/get_imm_post_dominators_map.h"
#include "utils/graph/digraph/algorithms/get_node_with_greatest_topo_rank.h"
#include "utils/graph/digraph/algorithms/get_post_dominators.h"
#include "utils/graph/digraph/algorithms/get_post_dominators_map.h"
#include "utils/optional.h"

namespace FlexFlow {

std::optional<Node> get_imm_post_dominator(DiGraphView const &g,
                                           Node const &n) {
  return get_imm_post_dominators_map(g).at(n);
}

std::optional<Node>
    get_imm_post_dominator(DiGraphView const &g,
                           std::unordered_set<Node> const &nodes) {

  if (nodes.empty()) {
    throw mk_runtime_error("Cannot get imm_post_dominator of no nodes");
  }

  if (nodes.size() == 1) {
    return get_imm_post_dominator(g, get_only(nodes));
  }

  Node contracted_node = get_first(nodes);
  std::unordered_map<Node, Node> contraction =
      generate_map(nodes, [&](Node const &) { return contracted_node; });
  return get_imm_post_dominator(apply_contraction(g, contraction),
                                contracted_node);
}

} // namespace FlexFlow

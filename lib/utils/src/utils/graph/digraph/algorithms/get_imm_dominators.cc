#include "utils/graph/digraph/algorithms/get_imm_dominators.h"
#include "utils/graph/digraph/algorithms/get_node_with_greatest_topo_rank.h"
#include "utils/graph/digraph/algorithms/get_dominators.h"
#include "utils/containers/get_only.h"

namespace FlexFlow {

std::unordered_map<Node, std::optional<Node>>
    get_imm_dominators(DiGraphView const &g) {

  std::unordered_map<Node, std::optional<Node>> result;
  for (auto const &kv : get_dominators(g)) {
    Node node = kv.first;
    std::unordered_set<Node> node_dominators = kv.second;

    assert(node_dominators.size() >= 1);

    // a node cannot immediately dominate itself
    if (node_dominators.size() == 1) {
      assert(get_only(node_dominators) == node);
      result[node] = std::nullopt;
    } else {
      node_dominators.erase(node);
      result[node] = get_node_with_greatest_topo_rank(node_dominators, g);
    }
  }
  return result;
}

} // namespace FlexFlow

#include "utils/graph/digraph/algorithms/apply_contraction.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/keys.h"
#include "utils/graph/digraph/algorithms/contract_node.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

DiGraphView apply_contraction(DiGraphView const &g,
                              std::unordered_map<Node, Node> const &nodes) {
  auto get_dst = [&](Node const &src) {
    Node result = src;
    while (contains_key(nodes, result)) {
      if (nodes.at(result) == result) {
        break;
      }

      result = nodes.at(result);
    }
    return result;
  };

  DiGraphView contractedView = g;
  for (Node const &src : keys(nodes)) {
    Node dst = get_dst(src);
    if (src != dst) {
      contractedView = contract_node(contractedView, src, dst);
    }
  }
  return contractedView;
}

} // namespace FlexFlow

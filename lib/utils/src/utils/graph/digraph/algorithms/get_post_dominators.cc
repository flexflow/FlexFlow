#include "utils/graph/digraph/algorithms/get_post_dominators.h"
#include "utils/graph/digraph/algorithms/get_dominators.h"
#include "utils/graph/digraph/algorithms.h"

namespace FlexFlow {

std::unordered_set<Node>
    get_post_dominators(DiGraphView const &g, Node const &n) {
  return get_post_dominators(g).at(n);
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_post_dominators(DiGraphView const &g) {
  return get_dominators(flipped(g));
}

} // namespace FlexFlow

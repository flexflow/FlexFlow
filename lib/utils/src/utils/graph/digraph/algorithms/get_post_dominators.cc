#include "utils/graph/digraph/algorithms/get_post_dominators.h"
#include "utils/graph/digraph/algorithms/flipped.h"
#include "utils/graph/digraph/algorithms/get_post_dominators_map.h"

namespace FlexFlow {

std::unordered_set<Node>
    get_post_dominators(DiGraphView const &g, Node const &n) {
  return get_post_dominators_map(g).at(n);
}

} // namespace FlexFlow

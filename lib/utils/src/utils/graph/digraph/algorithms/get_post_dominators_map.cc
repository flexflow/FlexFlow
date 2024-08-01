#include "utils/graph/digraph/algorithms/get_post_dominators_map.h"
#include "utils/graph/digraph/algorithms/flipped.h"
#include "utils/graph/digraph/algorithms/get_dominators_map.h"

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<Node>>
    get_post_dominators_map(DiGraphView const &g) {
  return get_dominators_map(flipped(g));
}

} // namespace FlexFlow

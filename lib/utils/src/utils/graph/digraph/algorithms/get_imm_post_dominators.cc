#include "utils/graph/digraph/algorithms/get_imm_post_dominators.h"
#include "utils/graph/digraph/algorithms/get_imm_dominators.h"
#include "utils/graph/digraph/algorithms.h"

namespace FlexFlow {

std::unordered_map<Node, std::optional<Node>>
    get_imm_post_dominators(DiGraphView const &g) {
  return get_imm_dominators(flipped(g));
}

} // namespace FlexFlow


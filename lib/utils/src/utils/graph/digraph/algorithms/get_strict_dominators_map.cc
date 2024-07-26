#include "utils/graph/digraph/algorithms/get_strict_dominators_map.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms/get_dominators_map.h"

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<Node>>
    get_strict_dominators_map(DiGraphView const &g) {
  return transform(get_dominators_map(g),
                   [](Node const &n, std::unordered_set<Node> const &doms) {
                     std::unordered_set<Node> result = doms;
                     result.erase(n);
                     return std::make_pair(n, result);
                   });
}

} // namespace FlexFlow

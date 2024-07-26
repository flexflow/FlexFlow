#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/algorithms/flipped.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<Node>>
    get_successors(DiGraphView const &g) {
  return get_predecessors(flipped(g));
}

std::unordered_set<Node> get_successors(DiGraphView const &g, Node const &n) {
  return get_predecessors(flipped(g), n);
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_successors(DiGraphView const &g, std::unordered_set<Node> const &ns) {
  return get_predecessors(flipped(g), ns);
}

} // namespace FlexFlow

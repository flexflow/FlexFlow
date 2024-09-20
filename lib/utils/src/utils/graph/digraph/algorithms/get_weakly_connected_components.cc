#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/undirected/algorithms/get_connected_components.h"

namespace FlexFlow {

std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(DiGraphView const &g) {
  return get_connected_components(as_undirected(g));
}

} // namespace FlexFlow

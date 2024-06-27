#include "utils/graph/simple_sp_ization.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/multidigraph.h"
#include "utils/graph/serialparallel.h"
#include <cassert>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

SerialParallelDecomposition simple_sp_ization_unchecked(DiGraphView const &g) {

  std::unordered_map<Node, int> layer_map = get_longest_path_lengths(g);
  std::map<int, std::unordered_set<Node>> layers;
  for (Node const &node : get_nodes(g)) {
    int layer_num = layer_map[node];
    layers[layer_num].insert(node);
  }

  Serial sp;

  for (auto const &[_, nodes] : layers) {
    Parallel layer;
    for (Node const &node : nodes) {
      layer.children.push_back({node});
    }
    sp.children.push_back(layer);
  }
  return sp;
}

SerialParallelDecomposition simple_sp_ization(DiGraph g) {
  assert(is_acyclic(g));
  assert(has_single_source(g));
  assert(has_single_sink(g));
  return simple_sp_ization_unchecked(g);
}

}; // namespace FlexFlow

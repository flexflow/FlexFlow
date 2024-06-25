#include "utils/graph/simple_sp_ization.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/multidigraph.h"
#include <cassert>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

DiGraph simple_sp_ization(DiGraph g) {

  std::unordered_set<Node> sources = get_sources(g);
  std::unordered_map<Node, int> layer_map;
  std::queue<Node> to_be_processed;
  for (Node const &source : sources) {
    layer_map[source] = 0;
    to_be_processed.push(source);
  }
  // Find the layer number for every node in g
  while (!to_be_processed.empty()) {
    Node predecessor = to_be_processed.front();
    to_be_processed.pop();
    for (Node const &successor : get_successors(g, predecessor)) {
      layer_map[successor] = layer_map[predecessor] + 1;
      to_be_processed.push(successor);
    }
  }

  std::map<int, std::unordered_set<Node>> layers;
  for (Node const &node : get_nodes(g)) {
    int layer_num = layer_map[node];
    layers[layer_num].insert(node);
  }

  DiGraph sp = DiGraph::create<AdjacencyDiGraph>();

  // Generate the SP-ized graph one layer at a time
  std::vector<Node> predecessors = add_nodes(sp, layers[0].size());
  for (auto const &[layer, _] : layers) {
    if (layers.count(layer + 1)) { // Skip if last layer
      std::vector<Node> successors = add_nodes(sp, layers.at(layer + 1).size());
      Node barrier_node = add_nodes(sp, 1)[0];
      for (auto const &p : predecessors) {
        sp.add_edge({p, barrier_node});
      }
      for (auto const &s : successors) {
        sp.add_edge({barrier_node, s});
      }
      predecessors = successors;
    }
  }

  return sp;
}
}; // namespace FlexFlow

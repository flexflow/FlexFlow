#include "pcg/parallel_computation_graph.h"
#include "utils/graph/algorithms.h"

namespace FlexFlow {

bool operator==(ParallelComputationGraph const &lhs, ParallelComputationGraph const &rhs) {
  return std::hash<ParallelComputationGraph>{}(lhs) == std::hash<ParallelComputationGraph>{}(rhs);
}

}

namespace std {

size_t hash<FlexFlow::ParallelComputationGraph>::operator()(FlexFlow::ParallelComputationGraph const &g) const {
  using namespace FlexFlow;

  size_t h = 0;

  std::vector<Node> ordered_nodes = get_topological_ordering(g.value());
  hash_combine(h, ordered_nodes.size());

  std::unordered_map<Node, int> node_index;
  for (int i = 0; i < ordered_nodes.size(); ++i) {
    node_index[ordered_nodes[i]] = i;
    hash_combine(h, g.value().at(ordered_nodes[i]));
  }

  for (MultiDiEdge const &edge : get_edges(g.value())) {
    hash_combine(h, node_index.at(edge.src));
    hash_combine(h, node_index.at(edge.dst));
    hash_combine(h, g.value().at(edge));
  }

  return h;
}

}
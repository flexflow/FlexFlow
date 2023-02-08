#include "utils/graph/adjacency_digraph.h"

namespace FlexFlow {
namespace utils {
namespace graph {
namespace digraph {

Node AdjacencyDiGraph::add_node() {
  Node node{this->next_node_idx};
  adjacency[node];
  this->next_node_idx++;
  return node;
}

void AdjacencyDiGraph::add_edge(Edge const &e) {
  this->adjacency.at(e.dst);
  this->adjacency.at(e.src).insert(e.dst);
}

std::unordered_set<Edge> AdjacencyDiGraph::query_edges(EdgeQuery const &query) const {
  std::unordered_set<Edge> result;
  for (auto const &kv : this->adjacency) {
    Node src = kv.first;
    if (!query.srcs.has_value() || query.srcs->find(src) != query.srcs->end()) {
      for (Node const &dst: kv.second) {
        if (!query.dsts.has_value() || query.dsts->find(dst) != query.dsts->end()) {
          result.insert({src, dst});
        }
      }
    }
  }
  return result;
}

std::unordered_set<Node> AdjacencyDiGraph::query_nodes(NodeQuery const &query) const {
  std::unordered_set<Node> result;
  for (auto const &kv : this->adjacency) {
    if (!query.nodes.has_value() || query.nodes->find(kv.first) != query.nodes->end()) {
      result.insert(kv.first);
    }
  }
  return result;
}

}
}
}
}

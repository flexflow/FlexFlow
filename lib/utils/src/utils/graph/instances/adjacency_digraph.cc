#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/containers/keys.h"
#include <cassert>

namespace FlexFlow {

AdjacencyDiGraph::AdjacencyDiGraph() {}

AdjacencyDiGraph::AdjacencyDiGraph(NodeSource const &node_source, std::unordered_map<Node, std::unordered_set<Node>> const &adjacency)
  : node_source(node_source), adjacency(adjacency) {}

AdjacencyDiGraph *AdjacencyDiGraph::clone() const {
  return new AdjacencyDiGraph(this->node_source, this->adjacency);
}

Node AdjacencyDiGraph::add_node() {
  Node new_node = this->node_source.new_node();
  this->adjacency[new_node];
  return new_node;
}

void AdjacencyDiGraph::add_node_unsafe(Node const &node) {
  this->adjacency[node];
}

void AdjacencyDiGraph::remove_node_unsafe(Node const &n) {
  auto iter = this->adjacency.find(n);
  if (iter != this->adjacency.end()) {
    this->adjacency.erase(iter);
  }
}

void AdjacencyDiGraph::add_edge(DirectedEdge const &e) {
  this->adjacency.at(e.dst);
  this->adjacency.at(e.src).insert(e.dst);
}

void AdjacencyDiGraph::remove_edge(DirectedEdge const &e) {
  std::unordered_set<Node> &m = this->adjacency.at(e.src);
  auto iter = m.find(e.dst);
  if (iter != m.end()) {
    m.erase(iter);
  }
}

std::unordered_set<DirectedEdge>
    AdjacencyDiGraph::query_edges(DirectedEdgeQuery const &query) const {
  std::unordered_set<DirectedEdge> result;
  for (auto const &src_kv : query_keys(query.srcs, this->adjacency)) {
    for (auto const &dst : apply_query(query.dsts, src_kv.second)) {
      result.insert(DirectedEdge{src_kv.first, dst});
    }
  }
  return result;
}

std::unordered_set<Node>
    AdjacencyDiGraph::query_nodes(NodeQuery const &query) const {
  return apply_query(query.nodes, keys(this->adjacency));
}

// bool AdjacencyDiGraph::operator==(AdjacencyDiGraph const &other) const {
//   bool result = this->adjacency == other.adjacency;
//   return result;
// }
//
// bool AdjacencyDiGraph::operator!=(AdjacencyDiGraph const &other) const {
//   return (this->adjacency != other.adjacency);
// }

} // namespace FlexFlow

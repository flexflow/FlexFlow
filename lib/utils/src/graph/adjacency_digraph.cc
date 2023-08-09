#include "utils/graph/adjacency_digraph.h"
#include <cassert>

namespace FlexFlow {

Node AdjacencyDiGraph::add_node() {
  Node node{this->next_node_idx};
  adjacency[node];
  this->next_node_idx++;
  return node;
}

void AdjacencyDiGraph::add_node_unsafe(Node const &node) {
  adjacency[node];
  this->next_node_idx = std::max(this->next_node_idx, node.value() + 1);
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
      result.insert({src_kv.first, dst});
    }
  }
  return result;
}

std::unordered_set<Node>
    AdjacencyDiGraph::query_nodes(NodeQuery const &query) const {
  return apply_query(query.nodes, keys(this->adjacency));
}

bool AdjacencyDiGraph::operator==(AdjacencyDiGraph const &other) const {
  bool result = this->adjacency == other.adjacency;
  if (result) {
    assert(this->next_node_idx == other.next_node_idx);
  }
  return result;
}

bool AdjacencyDiGraph::operator!=(AdjacencyDiGraph const &other) const {
  return (this->adjacency != other.adjacency);
}

} // namespace FlexFlow

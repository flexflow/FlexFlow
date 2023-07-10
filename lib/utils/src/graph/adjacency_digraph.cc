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
  for (auto const &kv : this->adjacency) {
    Node src = kv.first;
    if (!query.srcs.has_value() || query.srcs->find(src) != query.srcs->end()) {
      for (Node const &dst : kv.second) {
        if (!query.dsts.has_value() ||
            query.dsts->find(dst) != query.dsts->end()) {
          result.insert({src, dst});
        }
      }
    }
  }
  return result;
}

std::unordered_set<Node>
    AdjacencyDiGraph::query_nodes(NodeQuery const &query) const {
  std::unordered_set<Node> result;
  for (auto const &kv : this->adjacency) {
    if (!query.nodes.has_value() ||
        query.nodes->find(kv.first) != query.nodes->end()) {
      result.insert(kv.first);
    }
  }
  return result;
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

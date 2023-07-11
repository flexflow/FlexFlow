#include "utils/graph/adjacency_multidigraph.h"
#include "utils/containers.h"

namespace FlexFlow {

Node AdjacencyMultiDiGraph::add_node() {
  Node node{this->next_node_idx};
  adjacency[node];
  this->next_node_idx++;
  return node;
}

NodePort AdjacencyMultiDiGraph::add_node_port() {
  NodePort nodePort{this->next_node_port};
  this->next_node_port++;
  return nodePort;
}

void AdjacencyMultiDiGraph::add_node_port_unsafe(NodePort const &nodePort) {
  this->next_node_port = std::max(this->next_node_port, nodePort.value() + 1);
}

void AdjacencyMultiDiGraph::add_node_unsafe(Node const &node) {
  adjacency[node];
  this->next_node_idx = std::max(this->next_node_idx, node.value() + 1);
}

void AdjacencyMultiDiGraph::remove_node_unsafe(Node const &n) {
  this->adjacency.erase(n);
}

void AdjacencyMultiDiGraph::add_edge(MultiDiEdge const &e) {
  this->adjacency.at(e.dst);
  this->adjacency.at(e.src)[e.dst][e.srcIdx].insert(e.dstIdx);
}

void AdjacencyMultiDiGraph::remove_edge(MultiDiEdge const &e) {
  this->adjacency.at(e.src)[e.dst][e.srcIdx].erase(e.dstIdx);
}

std::unordered_set<MultiDiEdge>
    AdjacencyMultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  std::unordered_set<MultiDiEdge> result;
  for (auto const &kv : this->adjacency) {
    Node src = kv.first;
    if (!q.srcs.has_value() || contains(*q.srcs, src)) {
      for (auto const &kv2 : kv.second) {
        Node dst = kv2.first;
        if (!q.dsts.has_value() || contains(*q.dsts, dst)) {
          for (auto const &kv3 : kv2.second) {
            NodePort srcIdx = kv3.first;
            if (!q.srcIdxs.has_value() || contains(*q.srcIdxs, srcIdx)) {
              for (NodePort dstIdx : kv3.second) {
                if (!q.dstIdxs.has_value() || contains(*q.dstIdxs, dstIdx)) {
                  result.insert({src, dst, srcIdx, dstIdx});
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

std::unordered_set<Node>
    AdjacencyMultiDiGraph::query_nodes(NodeQuery const &query) const {
  std::unordered_set<Node> result;
  for (auto const &kv : this->adjacency) {
    if (!query.nodes.has_value() ||
        query.nodes->find(kv.first) != query.nodes->end()) {
      result.insert(kv.first);
    }
  }
  return result;
}

} // namespace FlexFlow

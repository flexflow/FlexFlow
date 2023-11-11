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
  auto nodePort = NodePort{this->next_node_port};
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
  /*
  this->adjacency.at(e.dst);  //has some bug 
  this->adjacency.at(e.src)[e.dst][e.src_idx].insert(e.dst_idx);
  this cause terminate called after throwing an instance of 'std::out_of_range'
  what():  _Map_base::at when we first meet e.dst
  */

  auto dst_it = this->adjacency.find(e.dst);
  if (dst_it == this->adjacency.end()) {
      dst_it = this->adjacency.insert({e.dst, {}}).first;
  }
  auto& src_map = this->adjacency[e.src];
  auto src_map_it = src_map.find(e.dst);
  if (src_map_it == src_map.end()) {
      src_map_it = src_map.insert({e.dst, {}}).first;
  }
  src_map_it->second[e.src_idx].insert(e.dst_idx);
}

void AdjacencyMultiDiGraph::remove_edge(MultiDiEdge const &e) {
  this->adjacency.at(e.src)[e.dst][e.src_idx].erase(e.dst_idx);
}

std::unordered_set<MultiDiEdge>
    AdjacencyMultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  std::unordered_set<MultiDiEdge> result;
  for (auto const &src_kv : query_keys(q.srcs, this->adjacency)) {
    for (auto const &dst_kv : query_keys(q.dsts, src_kv.second)) {
      for (auto const &srcIdx_kv : query_keys(q.srcIdxs, dst_kv.second)) {
        for (auto const &dstIdx : apply_query(q.dstIdxs, srcIdx_kv.second)) {
          result.insert(
              MultiDiEdge{dst_kv.first, dstIdx, src_kv.first, srcIdx_kv.first});
        }
      }
    }
  }
  return result;
}

std::unordered_set<Node>
    AdjacencyMultiDiGraph::query_nodes(NodeQuery const &query) const {
  return apply_query(query.nodes, keys(this->adjacency));
}

} // namespace FlexFlow

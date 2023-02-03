#include "utils/graph/adjacency_multidigraph.h"

using Node = FlexFlow::utils::graph::Node;
using namespace FlexFlow::utils::graph::multidigraph;

Node AdjacencyMultiDiGraph::add_node() {
  Node node{this->next_node_idx};
  adjacency[node];
  this->next_node_idx++;
  return node;
}

void AdjacencyMultiDiGraph::add_edge(Edge const &e) {
  this->adjacency.at(e.dst);
  this->adjacency.at(e.src)[e.dst][e.srcIdx].insert(e.dstIdx);
}

std::unordered_set<Edge> AdjacencyMultiDiGraph::query_edges(EdgeQuery const &q) const {
  std::unordered_set<Edge> result;
  for (auto const &kv : this->adjacency) {
    Node src = kv.first;
    if (!q.srcs.has_value() || q.srcs->find(src) != q.srcs->end()) {
      for (auto const &kv2 : kv.second) {
        Node dst = kv2.first;
        if (!q.dsts.has_value() || q.dsts->find(dst) != q.dsts->end()) {
          for (auto const &kv3 : kv2.second) {
            std::size_t srcIdx = kv3.first;
            if (!q.srcIdxs.has_value() || q.srcIdxs->find(srcIdx) != q.srcIdxs->end()) {
              for (std::size_t dstIdx : kv3.second) {
                if (!q.dstIdxs.has_value() || q.dstIdxs->find(dstIdx) != q.dstIdxs->end()) {
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

std::unordered_set<Node> AdjacencyMultiDiGraph::query_nodes(NodeQuery const &query) const {
  std::unordered_set<Node> result;
  for (auto const &kv : this->adjacency) {
    if (!query.nodes.has_value() || query.nodes->find(kv.first) != query.nodes->end()) {
      result.insert(kv.first);
    }
  }
  return result;
}

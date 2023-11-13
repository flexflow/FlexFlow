#include "utils/graph/adjacency_multidigraph.h"
#include "utils/containers.h"

namespace FlexFlow {

AdjacencyMultiDiGraph *AdjacencyMultiDiGraph::clone() const {
  return new AdjacencyMultiDiGraph(
      this->next_node_idx, this->next_node_port, this->adjacency);
}

AdjacencyMultiDiGraph::AdjacencyMultiDiGraph(std::size_t next_node_idx,
                                             std::size_t next_node_port,
                                             ContentsType const &adjacency)
    : next_node_idx(next_node_idx), next_node_port(next_node_port),
      adjacency(adjacency) {}

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
  std::cout << "e.dst:" << e.dst << ",e.dst_idx:" << e.dst_idx
            << ", e.src:" << e.src << ",e.src_idx" << e.src_idx << std::endl;
  /*auto dst_it = this->adjacency.find(e.dst);
  if (dst_it == this->adjacency.end()) {
      dst_it = this->adjacency.insert({e.dst, {}}).first;
  }
  auto& src_map = this->adjacency[e.src];
  auto src_map_it = src_map.find(e.src);
  if (src_map_it == src_map.end()) {
      src_map_it = src_map.insert({e.src, {}}).firs;
  }
  src_map_it->second[e.src_idx].insert(e.dst_idx);*/
  if (this->adjacency.count(e.dst) == 0) {
    this->adjacency.insert({e.dst, {}});
  } else {
    this->adjacency.at(e.dst);
  }
  // this->adjacency.at(e.src)[e.dst][e.src_idx].insert(e.dst_idx);
  if (this->adjacency.count(e.src) == 0) {
    this->adjacency.insert({e.src, {}});
  }
  if (this->adjacency.at(e.src).count(e.dst) == 0) {
    this->adjacency.at(e.src).insert({e.dst, {}});
  }
  if (this->adjacency.at(e.src).at(e.dst).count(e.src_idx) == 0) {
    this->adjacency.at(e.src)[e.dst].insert({e.src_idx, {e.dst_idx}});
  } else {
    this->adjacency.at(e.src)[e.dst][e.src_idx].insert(e.dst_idx);
  }
  // std::cout<<"this->adjacency.size():"<<this->adjacency.size()<<std::endl;
  for (auto const &entry1 : this->adjacency) {
    Node node1 = entry1.first;
    auto const &innerMap1 = entry1.second;
    if (innerMap1.size() > 0) {
      std::cout << "Node1: " << node1
                << ", InnerMap1 Size: " << innerMap1.size() << std::endl;
    }

    for (auto const &entry2 : innerMap1) {
      Node node2 = entry2.first;
      auto const &innerMap2 = entry2.second;
      if (innerMap2.size() > 0) {
        std::cout << "  Node2: " << node2
                  << ", InnerMap2 Size: " << innerMap2.size() << std::endl;
      }

      for (auto const &entry3 : innerMap2) {
        NodePort nodePort = entry3.first;
        auto const &nodePortSet = entry3.second;
        if (nodePortSet.size() > 0) {
          std::cout << "    NodePort: " << nodePort
                    << ", NodePortSet Size: " << nodePortSet.size()
                    << std::endl;
        }
        for (auto const &np : nodePortSet) {
          std::cout << "      NodePort: " << np << std::endl;
        }
      }
    }
  }
}

void AdjacencyMultiDiGraph::remove_edge(MultiDiEdge const &e) {
  this->adjacency.at(e.src)[e.dst][e.src_idx].erase(e.dst_idx);
}

// this has some bug, for example, for q, we only has the q.dsts, but don't have
// q.srcs how to handle the case when q doesn't hold
// src/dst/srcidx/dstidx(q.srcs is null),
// TODO:fix the corner case(q doesn't hold src/dst/srcidx/dstidx(q.srcs is
// null)) q.src is null, we return this->adjacency
std::unordered_set<MultiDiEdge>
    AdjacencyMultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  std::unordered_set<MultiDiEdge> result;
  std::cout << "1 query_keys(q.srcs, this->adjacency).size():"
            << query_keys(q.srcs, this->adjacency).size()
            << " and this->adjacency.size():" << this->adjacency.size()
            << std::endl;
  for (auto const &src_kv : query_keys(q.srcs, this->adjacency)) {
    std::cout << "1.5 query_keys(q.dsts, this->adjacency).size():"
              << query_keys(q.dsts, src_kv.second).size()
              << " and src_kv.second.size():" << src_kv.second.size()
              << ", src_kv.first:" << src_kv.first.value() << std::endl;
    for (auto const &dst_kv : query_keys(q.dsts, src_kv.second)) {
      std::cout << "2 query_keys(q.dsts, this->adjacency).size():"
                << query_keys(q.dsts, src_kv.second).size()
                << " and src_kv.second.size():" << src_kv.second.size()
                << std::endl;
      for (auto const &srcIdx_kv : query_keys(q.srcIdxs, dst_kv.second)) {
        std::cout << "3 query_keys(q.srcIdxs, dst_kv.second).size():"
                  << query_keys(q.srcIdxs, dst_kv.second).size()
                  << " and dst_kv.second.size():" << dst_kv.second.size()
                  << std::endl;

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

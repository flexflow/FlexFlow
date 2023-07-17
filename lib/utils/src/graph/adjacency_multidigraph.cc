#include "utils/graph/adjacency_multidigraph.h"
#include "utils/containers.h"

namespace FlexFlow {

Node AdjacencyMultiDiGraph::add_node() {
  Node node{this->next_node_idx};
  adjacency[node];
  std::cout << "add node " << node.value() << std::endl;
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
  // //std::cout<<"q.srcs:"<<q.srcs.value()<<" and q.dsts:"<<q.dsts.value()<< " and q.srcIdxs:"<<q.srcIdxs.value()<<" and q.dstIdxs:"<<q.dstIdxs.value()<<std::endl;
  // std::cout<<"AdjacencyMultiDiGraph::query_edges"<<std::endl;
  // for(auto kv : this->adjacency) {
  //   std::cout<<"this->adjacency node1:"<<kv.first.value()<<std::endl;
  //   for(auto node_nodeport: kv.second) {
  //     std::cout<<"this->adjacency node2:"<<node_nodeport.first.value()<<std::endl;
  //     for(auto nodeport_nodeport: node_nodeport.second) {
  //       std::cout<<"this->adjacency NodePort1:"<<nodeport_nodeport.first.value()<<std::endl;
  //       for(auto nodeport: nodeport_nodeport.second) {
  //         std::cout<<"this->adjacency NodePort2:"<<nodeport.value()<<std::endl;
  //       }
  //     }
  //   }
  //   std::cout<<"*********"<<std::endl;
  // }

  // auto src_kvs = query_keys(q.srcs, this->adjacency);
  // std::cout<<"x.size:"<<src_kvs.size()<<std::endl;
  // for(auto const &x:src_kvs) {
  //   std::cout<<"x.first:"<<x.first.value() <<" and x.second.size()"<<x.second.size()<<std::endl;
  // }
  for (auto const &src_kv : query_keys(q.srcs, this->adjacency)) {
    for (auto const &dst_kv : query_keys(q.dsts, src_kv.second)) {
      for (auto const &srcIdx_kv : query_keys(q.srcIdxs, dst_kv.second)) {
        for (auto const &dstIdx : apply_query(q.dstIdxs, srcIdx_kv.second)) {
          result.insert({src_kv.first, dst_kv.first, srcIdx_kv.first, dstIdx});
        }
      }
    }
  
  }
  std::cout<<"query_edges, result.size():"<<result.size()<<std::endl;
  return result;
}

std::unordered_set<Node>
    AdjacencyMultiDiGraph::query_nodes(NodeQuery const &query) const {
  return apply_query(query.nodes, keys(this->adjacency));
}

} // namespace FlexFlow

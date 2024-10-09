#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/keys.h"
#include "utils/exception.h"

namespace FlexFlow {

Node HashmapUndirectedGraph::add_node() {
  Node node = Node{this->next_node_idx};
  adjacency[node];
  this->next_node_idx++;
  return node;
}

void HashmapUndirectedGraph::add_node_unsafe(Node const &node) {
  adjacency[node];
  this->next_node_idx = std::max(this->next_node_idx, node.raw_uid + 1);
}

void HashmapUndirectedGraph::remove_node_unsafe(Node const &n) {
  this->adjacency.erase(n);
}

void HashmapUndirectedGraph::add_edge(UndirectedEdge const &e) {
  if (!contains_key(this->adjacency, e.bigger)) {
    throw mk_runtime_error(fmt::format(
        "Could not add edge connected to non-existent node {}", e.bigger));
  }
  if (!contains_key(this->adjacency, e.smaller)) {
    throw mk_runtime_error(fmt::format(
        "Could not add edge connected to non-existent node {}", e.smaller));
  }

  this->adjacency.at(e.bigger).insert(e.smaller);
  this->adjacency.at(e.smaller).insert(e.bigger);
}

void HashmapUndirectedGraph::remove_edge(UndirectedEdge const &e) {
  std::unordered_set<Node> &m = this->adjacency.at(e.bigger);
  m.erase(e.smaller);
  m.erase(e.bigger);
}

std::unordered_set<UndirectedEdge> HashmapUndirectedGraph::query_edges(
    UndirectedEdgeQuery const &query) const {
  std::unordered_set<UndirectedEdge> result;
  for (auto const &src_kv : query_keys(query.nodes, this->adjacency)) {
    for (auto const &dst : src_kv.second) {
      result.insert({src_kv.first, dst});
    }
  }
  return result;
}

std::unordered_set<Node>
    HashmapUndirectedGraph::query_nodes(NodeQuery const &query) const {
  return apply_query(query.nodes, keys(this->adjacency));
}

bool operator==(HashmapUndirectedGraph const &lhs,
                HashmapUndirectedGraph const &rhs) {
  bool result = lhs.adjacency == rhs.adjacency;
  if (result) {
    assert(lhs.next_node_idx == rhs.next_node_idx);
  }
  return result;
}

bool operator!=(HashmapUndirectedGraph const &lhs,
                HashmapUndirectedGraph const &rhs) {
  return (lhs.adjacency != rhs.adjacency);
}

} // namespace FlexFlow

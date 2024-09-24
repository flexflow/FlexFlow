#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/containers/contains.h"
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
  if (!contains_key(this->adjacency, e.endpoints.min())) {
    throw mk_runtime_error(
        "Could not add edge connected to non-existent node {}",
        e.endpoints.min());
  }
  if (!contains_key(this->adjacency, e.endpoints.max())) {
    throw mk_runtime_error(
        "Could not add edge connected to non-existent node {}",
        e.endpoints.max());
  }

  this->adjacency.at(e.endpoints.min()).insert(e.endpoints.max());
  this->adjacency.at(e.endpoints.max()).insert(e.endpoints.min());
}

void HashmapUndirectedGraph::remove_edge(UndirectedEdge const &e) {
  this->adjacency.at(e.endpoints.max()).erase(e.endpoints.min());
  this->adjacency.at(e.endpoints.min()).erase(e.endpoints.max());
}

std::unordered_set<UndirectedEdge> HashmapUndirectedGraph::query_edges(
    UndirectedEdgeQuery const &query) const {
  std::unordered_set<UndirectedEdge> result;
  std::unordered_set<Node> nodes =
      keys(query_keys(query.nodes, this->adjacency));
  for (auto const &[src, n] : query_keys(query.nodes, this->adjacency)) {
    for (auto const &dst : n) {
      result.insert(UndirectedEdge{{src, dst}});
    }
  }
  for (auto const &[src, n] :
       query_keys(query_set<Node>::matchall(), this->adjacency)) {
    for (auto const &dst : n) {
      if (contains(nodes, dst)) {
        result.insert(UndirectedEdge{{src, dst}});
      }
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

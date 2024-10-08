#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/extend.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/values.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/node/algorithms.h"
#include "utils/hash/unordered_set.h"

namespace FlexFlow {

AdjacencyMultiDiGraph::AdjacencyMultiDiGraph() {}

AdjacencyMultiDiGraph::AdjacencyMultiDiGraph(
    NodeSource const &node_source,
    MultiDiEdgeSource const &edge_source,
    std::unordered_map<
        Node,
        std::unordered_map<Node, std::unordered_set<MultiDiEdge>>> const
        &adjacency,
    std::unordered_map<MultiDiEdge, std::pair<Node, Node>> const &edge_nodes)
    : node_source(node_source), edge_source(edge_source), adjacency(adjacency),
      edge_nodes(edge_nodes) {}

Node AdjacencyMultiDiGraph::add_node() {
  Node new_node = this->node_source.new_node();
  std::unordered_set<Node> all_nodes =
      set_union(keys(this->adjacency), {new_node});
  this->adjacency[new_node] = generate_map(all_nodes, [](Node const &) {
    return std::unordered_set<MultiDiEdge>{};
  });

  for (Node const &n : all_nodes) {
    this->adjacency.at(n)[new_node] = {};
  }
  return new_node;
}

MultiDiEdge AdjacencyMultiDiGraph::add_edge(Node const &src, Node const &dst) {
  assert(contains_key(this->adjacency, src));
  assert(contains_key(this->adjacency, dst));
  MultiDiEdge new_edge = this->edge_source.new_multidiedge();
  this->adjacency.at(src).at(dst).insert(new_edge);
  this->edge_nodes.insert({new_edge, {src, dst}});
  return new_edge;
}

void AdjacencyMultiDiGraph::remove_node(Node const &n) {
  assert(contains_key(this->adjacency, n));

  std::unordered_set<MultiDiEdge> outgoing =
      set_union(values(this->adjacency.at(n)));
  std::unordered_set<MultiDiEdge> incoming;
  for (auto const &[k, v] : this->adjacency) {
    if (k != n) {
      extend(incoming, v.at(n));
    }
  }

  for (MultiDiEdge const &e : set_union(outgoing, incoming)) {
    this->edge_nodes.erase(e);
  }

  this->adjacency.erase(n);
  for (auto &[k, v] : this->adjacency) {
    v.erase(n);
  }
}

void AdjacencyMultiDiGraph::remove_edge(MultiDiEdge const &e) {
  assert(contains_key(this->edge_nodes, e));
  auto [src, dst] = this->edge_nodes.at(e);
  this->edge_nodes.erase(e);
  this->adjacency.at(src).at(dst).erase(e);
}

std::unordered_set<Node>
    AdjacencyMultiDiGraph::query_nodes(NodeQuery const &q) const {
  return apply_query(q.nodes, keys(this->adjacency));
}

std::unordered_set<MultiDiEdge>
    AdjacencyMultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  std::unordered_set<MultiDiEdge> result;

  std::unordered_set<Node> srcs = apply_query(q.srcs, keys(this->adjacency));
  std::unordered_set<Node> dsts = apply_query(q.dsts, keys(this->adjacency));
  for (Node const &src : srcs) {
    for (Node const &dst : dsts) {
      extend(result, this->adjacency.at(src).at(dst));
    }
  }

  return result;
}

Node AdjacencyMultiDiGraph::get_multidiedge_src(MultiDiEdge const &e) const {
  return this->edge_nodes.at(e).first;
}

Node AdjacencyMultiDiGraph::get_multidiedge_dst(MultiDiEdge const &e) const {
  return this->edge_nodes.at(e).second;
}

void AdjacencyMultiDiGraph::inplace_materialize_from(
    MultiDiGraphView const &g) {
  std::unordered_set<Node> nodes = get_nodes(g);
  std::unordered_set<MultiDiEdge> edges = get_edges(g);

  this->adjacency = generate_map(nodes, [&](Node const &) {
    return generate_map(
        nodes, [&](Node const &) { return std::unordered_set<MultiDiEdge>{}; });
  });
  this->edge_nodes.clear();

  for (MultiDiEdge const &e : edges) {
    Node src = g.get_multidiedge_src(e);
    Node dst = g.get_multidiedge_dst(e);
    this->adjacency.at(src).at(dst).insert(e);
    this->edge_nodes.insert({e, {src, dst}});
  }
}

AdjacencyMultiDiGraph *AdjacencyMultiDiGraph::clone() const {
  return new AdjacencyMultiDiGraph(
      this->node_source, this->edge_source, this->adjacency, this->edge_nodes);
}

} // namespace FlexFlow

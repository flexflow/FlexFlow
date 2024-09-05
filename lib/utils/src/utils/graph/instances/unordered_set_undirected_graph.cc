#include "utils/graph/instances/unordered_set_undirected_graph.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/undirected/undirected_edge_query.h"

namespace FlexFlow {

UnorderedSetUndirectedGraph::UnorderedSetUndirectedGraph() { }

UnorderedSetUndirectedGraph::UnorderedSetUndirectedGraph(NodeSource const &node_source,
                                                         std::unordered_set<Node> const &nodes,
                                                         std::unordered_set<UndirectedEdge> const &edges)
  : node_source(node_source), nodes(nodes), edges(edges)
{ }

Node UnorderedSetUndirectedGraph::add_node() {
  Node new_node = this->node_source.new_node();
  this->nodes.insert(new_node);
  return new_node;
}

void UnorderedSetUndirectedGraph::add_node_unsafe(Node const &n) {
  this->nodes.insert(n);
}

void UnorderedSetUndirectedGraph::remove_node_unsafe(Node const &n) {
  this->nodes.erase(n);
}

void UnorderedSetUndirectedGraph::add_edge(UndirectedEdge const &e) {
  assert (contains(this->nodes, e.bigger));
  assert (contains(this->nodes, e.smaller));
  this->edges.insert(e);
}

void UnorderedSetUndirectedGraph::remove_edge(UndirectedEdge const &e) {
  this->edges.erase(e);
}

std::unordered_set<Node> UnorderedSetUndirectedGraph::query_nodes(NodeQuery const &q) const {
  return apply_node_query(q, this->nodes);
}

std::unordered_set<UndirectedEdge> UnorderedSetUndirectedGraph::query_edges(UndirectedEdgeQuery const &q) const {
  return filter(this->edges, [&](UndirectedEdge const &e) { return matches_edge(q, e); });
}

UnorderedSetUndirectedGraph *UnorderedSetUndirectedGraph::clone() const {
  return new UnorderedSetUndirectedGraph{
    this->node_source,
    this->nodes, 
    this->edges,
  };
}

} // namespace FlexFlow

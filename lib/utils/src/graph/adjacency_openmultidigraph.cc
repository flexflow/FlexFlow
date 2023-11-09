#include "utils/graph/adjacency_openmultidigraph.h"

namespace FlexFlow {

void AdjacencyInputEdges::add_edge(InputMultiDiEdge const &e) {
  adj[e.dst][e.dst_idx].insert(e.uid);
}

void AdjacencyInputEdges::remove_edge(InputMultiDiEdge const &e) {
  adj[e.dst][e.dst_idx].erase(e.uid);
}

std::unordered_set<InputMultiDiEdge>
    AdjacencyInputEdges::query_edges(InputMultiDiEdgeQuery const &q) const {
  std::unordered_set<InputMultiDiEdge> result;
  for (auto const &[dst, dst_v] : query_keys(q.dsts, adj)) {
    for (auto const &[dst_idx, dst_idx_v] : query_keys(q.dstIdxs, dst_v)) {
      for (auto const &uid : dst_idx_v) {
        result.insert({dst, dst_idx, uid});
      }
    }
  }
  return result;
}

void AdjacencyOutputEdges::add_edge(OutputMultiDiEdge const &e) {
  adj[e.src][e.src_idx].insert(e.uid);
}

void AdjacencyOutputEdges::remove_edge(OutputMultiDiEdge const &e) {
  adj[e.src][e.src_idx].erase(e.uid);
}

std::unordered_set<OutputMultiDiEdge>
    AdjacencyOutputEdges::query_edges(OutputMultiDiEdgeQuery const &q) const {
  std::unordered_set<OutputMultiDiEdge> result;
  for (auto const &[src, src_v] : query_keys(q.srcs, adj)) {
    for (auto const &[src_idx, src_idx_v] : query_keys(q.srcIdxs, src_v)) {
      for (auto const &uid : src_idx_v) {
        result.insert({src, src_idx, uid});
      }
    }
  }
  return result;
}

std::unordered_set<Node>
    AdjacencyOpenMultiDiGraph::query_nodes(NodeQuery const &q) const {
  return closed_graph.query_nodes(q);
}

std::unordered_set<OpenMultiDiEdge> AdjacencyOpenMultiDiGraph::query_edges(
    OpenMultiDiEdgeQuery const &q) const {
  std::unordered_set<OpenMultiDiEdge> result;
  for (InputMultiDiEdge const &e : inputs.query_edges(q.input_edge_query)) {
    result.insert(e);
  }
  for (MultiDiEdge const &e : closed_graph.query_edges(q.standard_edge_query)) {
    result.insert(e);
  }
  for (OutputMultiDiEdge const &e : outputs.query_edges(q.output_edge_query)) {
    result.insert(e);
  }
  return result;
}

Node AdjacencyOpenMultiDiGraph::add_node() {
  return closed_graph.add_node();
}

NodePort AdjacencyOpenMultiDiGraph::add_node_port() {
  return closed_graph.add_node_port();
}

void AdjacencyOpenMultiDiGraph::add_node_unsafe(Node const &node) {
  closed_graph.add_node_unsafe(node);
}

void AdjacencyOpenMultiDiGraph::remove_node_unsafe(Node const &node) {
  closed_graph.remove_node_unsafe(node);
}

struct AddEdgeFunctor {
  AdjacencyInputEdges &inputs;
  AdjacencyOutputEdges &outputs;
  AdjacencyMultiDiGraph &closed_graph;

  template <typename T>
  void operator()(T const &e) {
    add_edge(e);
  }

  void add_edge(InputMultiDiEdge const &e) {
    inputs.add_edge(e);
  }

  void add_edge(OutputMultiDiEdge const &e) {
    outputs.add_edge(e);
  }

  void add_edge(MultiDiEdge const &e) {
    closed_graph.add_edge(e);
  }
};

void AdjacencyOpenMultiDiGraph::add_edge(OpenMultiDiEdge const &e) {
  visit(AddEdgeFunctor{inputs, outputs, closed_graph}, e);
}

struct RemoveEdgeFunctor {
  AdjacencyInputEdges &inputs;
  AdjacencyOutputEdges &outputs;
  AdjacencyMultiDiGraph &closed_graph;

  template <typename T>
  void operator()(T const &e) {
    remove_edge(e);
  }

  void remove_edge(InputMultiDiEdge const &e) {
    inputs.remove_edge(e);
  }

  void remove_edge(OutputMultiDiEdge const &e) {
    outputs.remove_edge(e);
  }

  void remove_edge(MultiDiEdge const &e) {
    closed_graph.remove_edge(e);
  }
};

void AdjacencyOpenMultiDiGraph::remove_edge(OpenMultiDiEdge const &e) {
  visit(RemoveEdgeFunctor{inputs, outputs, closed_graph}, e);
}

AdjacencyOpenMultiDiGraph::AdjacencyOpenMultiDiGraph(
    AdjacencyMultiDiGraph const &g,
    AdjacencyInputEdges const &inputs,
    AdjacencyOutputEdges const &outputs)
    : closed_graph(g.next_node_idx, g.next_node_port, g.adjacency),
      inputs(inputs), outputs(outputs) {}

AdjacencyOpenMultiDiGraph *AdjacencyOpenMultiDiGraph::clone() const {
  NOT_IMPLEMENTED(); // TODO
  // return new AdjacencyOpenMultiDiGraph(closed_graph, inputs, outputs);
}

} // namespace FlexFlow

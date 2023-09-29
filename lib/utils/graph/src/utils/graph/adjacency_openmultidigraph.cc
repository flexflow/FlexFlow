#include "utils/graph/adjacency_openmultidigraph.h"

namespace FlexFlow {

void AdjacencyInputs::add_edge(InputMultiDiEdge const &e) {
  adj[e.dst][e.dst_idx] = e.uid;
}

void AdjacencyInputs::remove_edge(InputMultiDiEdge const &e) {
  adj[e.dst][e.dst].erase(e.uid);
}

std::unordered_set<InputMultiDiEdge> AdjacencyInputs::query_edges(InputMultiDiEdgeQuery const &q) const {
  std::unordered_set<InputMultiDiEdge> result;
  for (auto const &[dst, dst_v] : query_keys(q.dsts, adj)) {
    for (auto const &[dst_idx, dst_idx_v] : query_keys(q.dstIdxs, dst_v)) {
      for (auto const &uid : dst_idx_v) {
        result.insert({dst, dst_idx, uid});
      }
    }
  }
}

void AdjacencyOutputs::add_edge(OutputMultiDiEdge const &e) {
  adj[e.src][e.src_idx] = e.uid;
}

void AdjacencyOutputs::remove_edge(OutputMultiDiEdge const &e) {
  adj[e.src][e.src_idx].erase(e.uid);
}

std::unordered_set<OutputMultiDiEdge> AdjacencyInputs::query_edges(OutputMultiDiEdgeQuery const &q) const {
  std::unordered_set<OutputMultiDiEdge> result;
  for (auto const &[src, src_v] : query_keys(q.srcs, adj)) {
    for (auto const &[src_idx, src_idx_v] : query_keys(q.srcIdxs, src_v)) {
      for (auto const &uid : src_idx_v) {
        result.insert({src, src_idx, uid});
      }
    }
  }
}

std::unordered_set<Node> AdjacencyOpenMultiDiGraph::query_nodes(NodeQuery const &q) const override {
  return closed_graph.query_nodes(q);
}

std::unordered_set<MultiDiEdge> AdjacencyOpenMultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const override {
  return closed_graph.query_edges(q);
}

std::unordered_set<OpenMultiDiEdge> AdjacencyOpenMultiDiGraph::query_edges(OpenMultiDiEdgeQuery const &q) const override {
  return set_union(
    {
      inputs.query_edges(q.input_edge_query),
      closed_graph.query_edges(q.standard_edge_query),
      outputs.query_edges(q.output_edge_query)
    }    
  );
}


void AdjacencyOpenMultiDiGraph::add_node(Node const &n) override {
  closed_graph.add_node(n);
}

struct AddEdgeFunctor {
  AdjacencyOpenMultiDiGraph* g;

  template <typename T>
  void operator()(OpenMultiDiEdge const &e) {
    add_edge(e);
  }

  void add_edge(InputMultiDiEdge const &e) {
    g->inputs.add_edge(e);
  }

  void add_edge(OutputMultiDiEdge const &e) {
    g->outputs.add_edge(e);
  }

  void add_edge(MultiDiEdge const &e) {
    g->closed_graph.add_edge(e);
  }
};

void AdjacencyOpenMultiDiGraph::add_edge(OpenMultiDiEdge const &e) override {
  visit(AddEdgeFunctor{this}, e);
}

struct RemoveEdgeFunctor {
  AdjacencyOpenMultiDiGraph* g;

  template <typename T>
  void operator()(OpenMultiDiEdge const &e) {
    remove_edge(e);
  }

  void remove_edge(InputMultiDiEdge const &e) {
    g->inputs.remove_edge(e);
  }

  void remove_edge(OutputMultiDiEdge const &e) {
    g->outputs.remove_edge(e);
  }

  void remove_edge(MultiDiEdge const &e) {
    g->closed_graph.remove_edge(e);
  }
};

void AdjacencyOpenMultiDiGraph::remove_edge(OpenMultiDiEdge const &e) override {
  visit(RemoveEdgeFunctor{this}, e);
}

AdjacencyOpenMultiDiGraph *AdjacencyOpenMultiDiGraph::clone() const override {
  return new AdjacencyOpenMultiDiGraph(*this);
}


}
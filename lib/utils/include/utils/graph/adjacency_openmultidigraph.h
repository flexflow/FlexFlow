#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_ADJACENCY_OPENMULTIDIGRAPH
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_ADJACENCY_OPENMULTIDIGRAPH

#include "open_graph_interfaces.h"
#include "adjacency_multidigraph.h"

namespace FlexFlow {

class AdjacencyInputs {
public:
  void add_edge(InputMultiDiEdge const &);
  void remove_edge(InputMultiDiEdge const &);
  std::unordered_set<InputMultiDiEdge> query_edges(InputMultiDiEdgeQuery const &) const;
private:
  using ContentsType = std::unordered_map<Node, std::unordered_map<NodePort, std::unordered_set<edge_uid_t>>>;
  ContentsType adj;
};

class AdjacencyOutputs {
public:
  void add_edge(OutputMultiDiEdge const &);
  void remove_edge(OutputMultiDiEdge const &);
  std::unordered_set<OutputMultiDiEdge> query_edges(OutputMultiDiEdgeQuery const &) const;
private:
  using ContentsType = std::unordered_map<Node, std::unordered_map<NodePort, std::unordered_set<edge_uid_t>>>;
  ContentsType adj;
};

class AdjacencyOpenMultiDiGraph : virtual public IOpenMultiDiGraph {
public:
  AdjacencyOpenMultiDiGraph() = default;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  // std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const override;

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const override;

  Node add_node() override;
  NodePort add_node_port() override;
  void add_node_unsafe(Node const &) override;
  void remove_node_unsafe(Node const &) override;
  void add_edge(OpenMultiDiEdge const &) override;
  void remove_edge(OpenMultiDiEdge const &) override;
  AdjacencyOpenMultiDiGraph *clone() const override;

private:
  AdjacencyOpenMultiDiGraph(AdjacencyMultiDiGraph const &g, AdjacencyInputs const &inputs, AdjacencyOutputs const &outputs);

  AdjacencyMultiDiGraph closed_graph;
  AdjacencyInputs inputs;
  AdjacencyOutputs outputs;
};

CHECK_NOT_ABSTRACT(AdjacencyOpenMultiDiGraph);

}


#endif

#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_ADJACENCY_OPENMULTIDIGRAPH
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_ADJACENCY_OPENMULTIDIGRAPH

#include "open_graph_interfaces.h"
#include "adjacency_multidigraph.h"

namespace FlexFlow {

class AdjancencyInputs {
public:
  void add_edge(InputMultiDiEdge const &);
  void remove_edge(InputMultiDiEdge const &);
  std::unordered_set<InputMultiDiEdge> query_edges(InputMultiDiEdgeQuery const &) const;
private:
  using ContentsType = std::unordered_map<Node, std::unordered_map<NodePort, edge_uid_t>>;
  ContentsType adj;
};

class AdjancencyOutputs {
public:
  void add_edge(OutputMultiDiEdge const &);
  void remove_edge(OutputMultiDiEdge const &);
  std::unordered_set<OutputMultiDiEdge> query_edges(OutputMultiDiEdgeQuery const &) const;
private:
  using ContentsType = std::unordered_map<Node, std::unordered_map<NodePort, edge_uid_t>>;
};

class AdjacencyOpenMultiDiGraph : virtual public IOpenMultiDiGraph {
public:
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const override;

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const override;

  void add_node(Node const &) override;
  void add_edge(OpenMultiDiEdge const &) override;
  void remove_edge(OpenMultiDiEdge const &) override;
  AdjacencyOpenMultiDiGraph *clone() const override;

private:
  AdjacencyMultiDiGraph closed_graph;
  AdjancencyInputs inputs;
  AdjancencyOutputs outputs;
};

}


#endif

#ifndef _FLEXFLOW_UTILS_ADJACENCY_MULTIGRAPH_H
#define _FLEXFLOW_UTILS_ADJACENCY_MULTIGRAPH_H 

#include "multidigraph.h"
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

class AdjacencyMultiDiGraph : public IMultiDiGraph {
public:
  Node add_node() override;
  void add_node_unsafe(Node const &) override;
  void remove_node_unsafe(Node const &) override;
  void add_edge(Edge const &) override;
  void remove_edge(Edge const &) override;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  void add_node(Node const &); //add node

  AdjacencyMultiDiGraph *clone() const override { 
    return new AdjacencyMultiDiGraph(this->next_node_idx, this->adjacency);
  }

  // //add my constructor
  // AdjacencyMultiDiGraph(AdjacencyMultiDiGraph const & g){
  //   this->next_node_idx = g.next_node_idx;
  //   this->adjacency = g.get_adjacency();
  // }


  AdjacencyMultiDiGraph(){
    next_node_idx = 0;
  }

  ~AdjacencyMultiDiGraph(){}

private:
  using ContentsType = 
    std::unordered_map<Node, 
      std::unordered_map<Node,
        std::unordered_map<std::size_t, std::unordered_set<std::size_t>>>>;

  AdjacencyMultiDiGraph(std::size_t next_node_idx_, ContentsType const & adjacency_)
  : next_node_idx(next_node_idx_), adjacency(adjacency_)
  {}

public:
  ContentsType get_adjacency(){
    return this->adjacency;
  } 
  
private:
  std::size_t next_node_idx = 0;
  ContentsType adjacency;
};

static_assert(is_rc_copy_virtual_compliant<AdjacencyMultiDiGraph>::value, RC_COPY_VIRTUAL_MSG);

}

#endif

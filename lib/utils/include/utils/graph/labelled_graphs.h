#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_GRAPHS_H

#include <memory>
#include "multidigraph.h"
#include <unordered_map>

namespace FlexFlow {

template <typename T>
struct LabelledMultiDiGraph {
public:
  IMultiDiGraph const &graph() const {
    return *this->base_graph;
  }
  IMultiDiGraph &graph() {
    return *this->base_graph;
  }

  Node add_node(T const &t) {
    Node n = this->base_graph->add_node();
    node_map.insert({ n, t });
    return n;
  }
  void add_edge(MultiDiEdge const &e) {
    return this->base_graph->add_edge(e);
  }

  T const &at(Node const &n) const {
    return this->node_map->at(n);
  }
public:
  std::unique_ptr<IMultiDiGraph> base_graph;
  std::unordered_map<Node, T> node_map;
};

}

#endif 

#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_GRAPHS_H

#include <memory>
#include "multidigraph.h"
#include <unordered_map>
#include "open_graphs.h"
#include "utils/unique.h"

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

template<typename NodeLabel, 
         typename EdgeLabel, 
         typename InputLabel = EdgeLabel, 
         typename OutputLabel = InputLabel>
struct LabelledOpenMultiDiGraph {
public:
  LabelledOpenMultiDiGraph(std::unique_ptr<IOpenMultiDiGraph> base)
    : base_graph(std::move(base))
  { }

  operator IOpenMultiDiGraph const &() const {
    return *this->base_graph;
  }

  Node add_node(NodeLabel const &t) {
    Node n = this->base_graph->add_node();
    node_map.insert({ n, t });
    return n;
  }

  void add_edge(InputMultiDiEdge const &e, InputLabel const &label) {
    this->base_graph->add_edge(e);
    this->input_map.insert({e, label});
  }

  void add_edge(MultiDiEdge const &e, EdgeLabel const &label) {
    this->base_graph->add_edge(e);
    this->edge_map.insert({e, label});
  }

  void add_edge(OutputMultiDiEdge const &e, OutputLabel const &label) {
    this->base_graph->add_edge(e);
    this->output_map.insert({e, label});
  }

  NodeLabel const &at(Node const &n) const {
    return this->node_map.at(n);
  }

  NodeLabel &at(Node const &n) {
    return this->node_map.at(n);
  }

  EdgeLabel const &at(MultiDiEdge const &e) const {
    return this->edge_map.at(e);
  }

  EdgeLabel &at(MultiDiEdge const &e) {
    return this->edge_map.at(e);
  }

  InputLabel const &at(InputMultiDiEdge const &e) const {
    return this->input_map.at(e);
  }

  InputLabel &at(InputMultiDiEdge const &e) {
    return this->input_map.at(e);
  }

  OutputLabel const &at(OutputMultiDiEdge const &e) const {
    return this->output_map.at(e);
  }

  OutputLabel &at(DownwardOpenMultiDiEdge const &e) {
    return this->output_map.at(e);
  }

  template <typename BaseImpl, 
            typename ...Args>
  static LabelledOpenMultiDiGraph create(Args &&... args) {
    return LabelledOpenMultiDiGraph(make_unique<BaseImpl>(std::forward<Args>(args)...));
  }
private:
  std::unique_ptr<IOpenMultiDiGraph> base_graph;
  std::unordered_map<Node, NodeLabel> node_map;
  std::unordered_map<MultiDiEdge, EdgeLabel> edge_map;
  std::unordered_map<InputMultiDiEdge, InputLabel> input_map;
  std::unordered_map<OutputMultiDiEdge, OutputLabel> output_map;
};
}

#endif 

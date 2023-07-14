#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_UNORDERED_LABELLED_GRAPHS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_UNORDERED_LABELLED_GRAPHS_H

#include "labelled_open_interfaces.h"
#include "node_labelled_interfaces.h"
#include "output_labelled_interfaces.h"
#include "standard_labelled_interfaces.h"
#include "utils/graph/open_graphs.h"

namespace FlexFlow {

template <typename NodeLabel>
struct UnorderedNodeLabelledMultiDiGraph
    : public INodeLabelledMultiDiGraph<NodeLabel>,
      protected MultiDiGraph {
public:
  UnorderedNodeLabelledMultiDiGraph() = delete;

  Node add_node(NodeLabel const &label) override {
    Node n = this->add_node();
    node_map.insert({n, label});
    return n;
  }

  NodeLabel &at(Node const &n) override {
    return this->node_map.at(n);
  }

  NodeLabel const &at(Node const &n) const override {
    return this->node_map.at(n);
  }

  using MultiDiGraph::query_edges;
  using MultiDiGraph::query_nodes;

private:
  std::unordered_map<Node, NodeLabel> node_map;
};

template <typename NodeLabel, typename EdgeLabel>
struct UnorderedLabelledMultiDiGraph
    : public ILabelledMultiDiGraph<NodeLabel, EdgeLabel>,
      public UnorderedNodeLabelledMultiDiGraph<NodeLabel> {
  void add_edge(MultiDiEdge const &e, EdgeLabel const &label) override {
    this->add_edge(e);
    edge_map.insert({e, label});
    return label;
  }

  EdgeLabel &at(MultiDiEdge const &n) override {
    return this->edge_map.at(n);
  }

  EdgeLabel const &at(MultiDiEdge const &n) const override {
    return this->edge_map.at(n);
  }

private:
  std::unordered_map<MultiDiEdge, EdgeLabel> edge_map;
};

MultiDiOutput get_output(MultiDiEdge const &e);

template <typename NodeLabel, typename OutputLabel>
struct UnorderedOutputLabelledMultiDiGraph
    : public IOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>,
      public UnorderedNodeLabelledMultiDiGraph<NodeLabel> {
public:
  void add_output(MultiDiOutput const &output,
                  OutputLabel const &label) override {
    this->output_map.insert({output, label});
  }

  void add_edge(MultiDiEdge const &e) override {
    MultiDiOutput output = get_output(e);
    if (!contains_key(this->output_map, output)) {
      throw mk_runtime_error("Could not find output {}", output);
    }
    this->add_edge(e);
  }

  void add_edge(MultiDiOutput const &output,
                MultiDiInput const &input) override {
    this->add_edge(MultiDiEdge{output.node, input.node, output.idx, input.idx});
  }

private:
  std::unordered_map<MultiDiOutput, OutputLabel> output_map;
};

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct UnorderedLabelledOpenMultiDiGraph
    : public ILabelledOpenMultiDiGraph<NodeLabel,
                                       EdgeLabel,
                                       InputLabel,
                                       OutputLabel>,
      public UnorderedLabelledMultiDiGraph<NodeLabel, EdgeLabel> {
public:
  void add_edge(InputMultiDiEdge const &e, InputLabel const &label) {
    this->add_edge(e);
    this->input_map.insert({e, label});
  }

  void add_edge(OutputMultiDiEdge const &e, OutputLabel const &label) {
    this->add_edge(e);
    this->output_map.insert({e, label});
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

private:
  OpenMultiDiGraph base_graph;
  std::unordered_map<InputMultiDiEdge, InputLabel> input_map;
  std::unordered_map<OutputMultiDiEdge, OutputLabel> output_map;
};

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_UNORDERED_LABELLED_GRAPHS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_UNORDERED_LABELLED_GRAPHS_H

#include "output_labelled_open_interfaces.h"
#include "unordered_label.h"
#include "utils/graph/adjacency_openmultidigraph.h"

namespace FlexFlow {

template <typename NodeLabel>
struct UnorderedNodeLabelledOpenMultiDiGraph
    : public INodeLabelledOpenMultiDiGraph<NodeLabel> {

  UnorderedNodeLabelledOpenMultiDiGraph()
      : g(OpenMultiDiGraph::create<AdjacencyOpenMultiDiGraph>()) {}

  Node add_node(NodeLabel const &l) override {
    Node node = g.add_node();
    this->node_labelling.add_label(node, l);
    return node;
  }

  NodePort add_node_port() override {
    return this->g.add_node_port();
  }

  NodeLabel const &at(Node const &n) const override {
    return this->node_labelling.get_label(n);
  }

  NodeLabel &at(Node const &n) override {
    return this->node_labelling.get_label(n);
  }

  void add_edge(OpenMultiDiEdge const &e) override {
    this->g.add_edge(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return g.query_nodes(q);
  }

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const override {
    return g.query_edges(q);
  }

  using INodeLabelledOpenMultiDiGraph<NodeLabel>::query_edges;

  UnorderedNodeLabelledOpenMultiDiGraph *clone() const override {
    return new UnorderedNodeLabelledOpenMultiDiGraph<NodeLabel>(g,
                                                                node_labelling);
  }

private:
  UnorderedNodeLabelledOpenMultiDiGraph(
      OpenMultiDiGraph const &g,
      UnorderedLabelling<Node, NodeLabel> const &node_labelling)
      : g(g), node_labelling(node_labelling) {}

  OpenMultiDiGraph g;
  UnorderedLabelling<Node, NodeLabel> node_labelling;
};
CHECK_NOT_ABSTRACT(UnorderedNodeLabelledOpenMultiDiGraph<test_types::hash_cmp>);

template <typename NodeLabel, typename OutputLabel>
struct UnorderedOutputLabelledMultiDiGraph
    : public IOutputLabelledMultiDiGraph<NodeLabel, OutputLabel> {

  UnorderedOutputLabelledMultiDiGraph()
      : g(MultiDiGraph::create<AdjacencyMultiDiGraph>()) {}

  OutputLabel const &at(MultiDiOutput const &i) const override {
    return this->output_labelling.get_label(i);
  }

  OutputLabel &at(MultiDiOutput const &i) override {
    return this->output_labelling.get_label(i);
  }

  Node add_node(NodeLabel const &l) override {
    Node node = g.add_node();
    this->node_labelling.add_label(node, l);
    return node;
  }

  NodePort add_node_port() override {
    return this->g.add_node_port();
  }

  NodeLabel const &at(Node const &n) const override {
    return this->node_labelling.get_label(n);
  }

  NodeLabel &at(Node const &n) override {
    return this->node_labelling.get_label(n);
  }

  void add_edge(MultiDiEdge const &e) override {
    this->g.add_edge(e);
  }

  void add_output(MultiDiOutput const &output,
                  OutputLabel const &label) override {
    this->output_labelling.add_label(output, label);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return g.query_nodes(q);
  }

  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &q) const override {
    return g.query_edges(q);
  }

  using IOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>::query_edges;

  UnorderedOutputLabelledMultiDiGraph *clone() const override {
    return new UnorderedOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>(
        g, node_labelling, output_labelling);
  }

private:
  UnorderedOutputLabelledMultiDiGraph(
      MultiDiGraph const &g,
      UnorderedLabelling<Node, NodeLabel> const &node_labelling,
      UnorderedLabelling<MultiDiOutput, OutputLabel> const &output_labelling)
      : g(g), node_labelling(node_labelling),
        output_labelling(output_labelling) {}

  MultiDiGraph g;
  UnorderedLabelling<Node, NodeLabel> node_labelling;
  UnorderedLabelling<MultiDiOutput, OutputLabel> output_labelling;
};
CHECK_NOT_ABSTRACT(UnorderedOutputLabelledMultiDiGraph<test_types::hash_cmp,
                                                       test_types::hash_cmp>);

template <typename NodeLabel, typename EdgeLabel>
struct UnorderedOutputLabelledOpenMultiDiGraph
    : public IOutputLabelledOpenMultiDiGraph<NodeLabel, EdgeLabel> {

  UnorderedOutputLabelledOpenMultiDiGraph()
      : g(OpenMultiDiGraph::create<AdjacencyOpenMultiDiGraph>()) {}

  EdgeLabel const &at(InputMultiDiEdge const &i) const override {
    return this->input_labelling.get_label(i);
  }

  EdgeLabel &at(InputMultiDiEdge const &i) override {
    return this->input_labelling.get_label(i);
  }

  EdgeLabel const &at(MultiDiOutput const &i) const override {
    return this->output_labelling.get_label(i);
  }

  EdgeLabel &at(MultiDiOutput const &i) override {
    return this->output_labelling.get_label(i);
  }

  Node add_node(NodeLabel const &l) override {
    Node node = g.add_node();
    this->node_labelling.add_label(node, l);
    return node;
  }

  NodePort add_node_port() override {
    return this->g.add_node_port();
  }

  NodeLabel const &at(Node const &n) const override {
    return this->node_labelling.get_label(n);
  }

  NodeLabel &at(Node const &n) override {
    return this->node_labelling.get_label(n);
  }

  void add_label(MultiDiOutput const &o, EdgeLabel const &l) override {
    this->output_labelling.add_label(o, l);
  }

  void add_label(InputMultiDiEdge const &i, EdgeLabel const &l) override {
    this->input_labelling.add_label(i, l);
  }

  void add_edge(OpenMultiDiEdge const &e) override {
    this->g.add_edge(e);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->g.query_nodes(q);
  }

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const override {
    return this->g.query_edges(q);
  }

  using IOutputLabelledOpenMultiDiGraph<NodeLabel, EdgeLabel>::query_edges;

  UnorderedOutputLabelledOpenMultiDiGraph *clone() const override {
    return new UnorderedOutputLabelledOpenMultiDiGraph<NodeLabel, EdgeLabel>(
        g, node_labelling, input_labelling, output_labelling);
  }

private:
  UnorderedOutputLabelledOpenMultiDiGraph(
      OpenMultiDiGraph const &g,
      UnorderedLabelling<Node, NodeLabel> const &node_labelling,
      UnorderedLabelling<InputMultiDiEdge, EdgeLabel> const &input_labelling,
      UnorderedLabelling<MultiDiOutput, EdgeLabel> const &output_labelling)
      : g(g), node_labelling(node_labelling), input_labelling(input_labelling),
        output_labelling(output_labelling) {}

  OpenMultiDiGraph g;
  UnorderedLabelling<Node, NodeLabel> node_labelling;
  UnorderedLabelling<InputMultiDiEdge, EdgeLabel> input_labelling;
  UnorderedLabelling<MultiDiOutput, EdgeLabel> output_labelling;
};
CHECK_NOT_ABSTRACT(
    UnorderedOutputLabelledOpenMultiDiGraph<test_types::hash_cmp,
                                            test_types::hash_cmp>);

} // namespace FlexFlow

#endif

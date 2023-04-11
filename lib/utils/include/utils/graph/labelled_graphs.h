#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_GRAPHS_H

#include <memory>
#include "multidigraph.h"
#include <unordered_map>
#include "open_graphs.h"
#include "utils/unique.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename NodeLabel>
struct NodeLabelledMultiDiGraph {
public:
  NodeLabelledMultiDiGraph() = delete;
  NodeLabelledMultiDiGraph(NodeLabelledMultiDiGraph const &) = default;
  NodeLabelledMultiDiGraph operator=(NodeLabelledMultiDiGraph const &) = default;

  operator MultiDiGraph const &() const {
    return this->base_graph;
  }

  Node add_node(NodeLabel const &label) {
    Node n = this->base_graph.add_node();
    node_map.insert({ n, label });
    return n;
  }

  NodeLabel &at(Node const &n) {
    return this->node_map.at(n);
  }

  NodeLabel const &at(Node const &n) const {
    return this->node_map.at(n);
  }
protected:
  MultiDiGraph base_graph;
private:
  std::unordered_map<Node, NodeLabel> node_map;
};

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraph : public NodeLabelledMultiDiGraph<NodeLabel> {
  void add_edge(MultiDiEdge const &e, EdgeLabel const &label) {
    this->base_graph.add_edge(e);
    edge_map.insert({ e, label });
    return label;
  }

  EdgeLabel &at(MultiDiEdge const &n) {
    return this->edge_map.at(n);
  }

  EdgeLabel const &at(MultiDiEdge const &n) const {
    return this->edge_map.at(n);
  }
private:
  std::unordered_map<MultiDiEdge, EdgeLabel> edge_map;
};

struct MultiDiOutput {
  MultiDiOutput(Node const &, size_t);

  Node node;
  size_t idx;
};

struct MultiDiInput {
  MultiDiInput(Node const &, size_t);

  Node node;
  size_t idx;
};

MultiDiOutput get_output(MultiDiEdge const &);
MultiDiInput get_input(MultiDiEdge const &);

template <typename NodeLabel, typename OutputLabel>
struct OutputLabelledMultiDiGraph : public NodeLabelledMultiDiGraph<NodeLabel> {
public:
  void add_output(MultiDiOutput const &output, OutputLabel const &label) {
    this->output_map.insert({ output, label });
  }

  void add_edge(MultiDiEdge const &e) {
    MultiDiOutput output = get_output(e);
    if (!contains_key(this->output_map, output)) {
      throw mk_runtime_error("Could not find output {}", output);
    }
    this->base_graph.add_edge(e);
  }

  void add_edge(MultiDiOutput const &output, MultiDiInput const &input) {
    this->add_edge(MultiDiEdge{output.node, input.node, output.idx, input.idx});
  }

  OutputLabel &at(MultiDiOutput const &output) {
    return this->output_map->at(output);
  }

  OutputLabel const &at(MultiDiOutput const &output) const {
    return this->output_map->at(output);
  }
private:
  std::unordered_map<MultiDiOutput, OutputLabel> output_map;
};

template<typename NodeLabel, 
         typename EdgeLabel, 
         typename InputLabel = EdgeLabel, 
         typename OutputLabel = InputLabel>
struct LabelledOpenMultiDiGraph {
public:
  LabelledOpenMultiDiGraph() = delete;
  LabelledOpenMultiDiGraph(LabelledOpenMultiDiGraph const &) = default;
  LabelledOpenMultiDiGraph& operator=(LabelledOpenMultiDiGraph const &) = default;

  operator OpenMultiDiGraph const &() const {
    return this->base_graph;
  }

  Node add_node(NodeLabel const &t) {
    Node n = this->base_graph.add_node();
    node_map.insert({ n, t });
    return n;
  }

  void add_edge(InputMultiDiEdge const &e, InputLabel const &label) {
    this->base_graph.add_edge(e);
    this->input_map.insert({e, label});
  }

  void add_edge(MultiDiEdge const &e, EdgeLabel const &label) {
    this->base_graph.add_edge(e);
    this->edge_map.insert({e, label});
  }

  void add_edge(OutputMultiDiEdge const &e, OutputLabel const &label) {
    this->base_graph.add_edge(e);
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

  template <typename BaseImpl>
  static 
  typename std::enable_if<std::is_base_of<IOpenMultiDiGraph, BaseImpl>::value, LabelledOpenMultiDiGraph>::type
  create() {
    return LabelledOpenMultiDiGraph(OpenMultiDiGraph::create<BaseImpl>());
  }

  friend void swap(LabelledOpenMultiDiGraph &lhs, LabelledOpenMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.base_graph, rhs.base_graph);
    swap(lhs.node_map, rhs.node_map);
    swap(lhs.edge_map, rhs.edge_map);
    swap(lhs.input_map, rhs.input_map);
    swap(lhs.output_map, rhs.output_map);
  }
private:
  LabelledOpenMultiDiGraph(OpenMultiDiGraph base)
    : base_graph(std::move(base))
  { }
private:
  OpenMultiDiGraph base_graph;
  std::unordered_map<Node, NodeLabel> node_map;
  std::unordered_map<MultiDiEdge, EdgeLabel> edge_map;
  std::unordered_map<InputMultiDiEdge, InputLabel> input_map;
  std::unordered_map<OutputMultiDiEdge, OutputLabel> output_map;
};

}

namespace fmt {

template <>
struct formatter<::FlexFlow::MultiDiOutput> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::MultiDiOutput const &x, FormatContext &ctx) const -> decltype(ctx.out()) {
    return formatter<std::string>::format(fmt::format("MultiDiOutput({}, {})", x.node, x.idx), ctx);
  }
};

template <>
struct formatter<::FlexFlow::MultiDiInput> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::MultiDiInput const &x, FormatContext &ctx) const -> decltype(ctx.out()) {
    return formatter<std::string>::format(fmt::format("MultiDiInput({}, {})", x.node, x.idx), ctx);
  }
};

}

#endif 

#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/containers.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.h"

namespace FlexFlow {

UnorderedSetDataflowGraph::UnorderedSetDataflowGraph() {}
UnorderedSetDataflowGraph::UnorderedSetDataflowGraph(
    NodeSource const &node_source,
    DataflowGraphInputSource const &graph_input_source,
    std::unordered_set<Node> const &nodes,
    std::unordered_set<OpenDataflowEdge> const &edges,
    std::unordered_set<DataflowOutput> const &outputs,
    std::unordered_set<DataflowGraphInput> const &graph_inputs)
    : node_source(node_source), graph_input_source(graph_input_source),
      nodes(nodes), edges(edges), outputs(outputs), graph_inputs(graph_inputs) {
}

NodeAddedResult UnorderedSetDataflowGraph::add_node(
    std::vector<DataflowOutput> const &inputs, int num_outputs) {
  std::vector<OpenDataflowValue> open_inputs = transform(
      inputs, [](DataflowOutput const &o) { return OpenDataflowValue{o}; });
  return this->add_node(open_inputs, num_outputs);
}

NodeAddedResult UnorderedSetDataflowGraph::add_node(
    std::vector<OpenDataflowValue> const &inputs, int num_outputs) {
  Node new_node = this->node_source.new_node();

  std::vector<DataflowOutput> new_outputs =
      transform(count(num_outputs), [&](int output_idx) {
        return DataflowOutput{new_node, output_idx};
      });

  this->add_node_unsafe(new_node, inputs, new_outputs);

  return NodeAddedResult{new_node, new_outputs};
}

DataflowGraphInput UnorderedSetDataflowGraph::add_input() {
  DataflowGraphInput new_graph_input =
      this->graph_input_source.new_dataflow_graph_input();

  this->graph_inputs.insert(new_graph_input);

  return new_graph_input;
}

std::unordered_set<Node>
    UnorderedSetDataflowGraph::query_nodes(NodeQuery const &q) const {
  return apply_query(q.nodes, this->nodes);
}

std::unordered_set<OpenDataflowEdge> UnorderedSetDataflowGraph::query_edges(
    OpenDataflowEdgeQuery const &q) const {
  return filter(this->edges, [&](OpenDataflowEdge const &e) {
    return open_dataflow_edge_query_includes(q, e);
  });
}

std::unordered_set<DataflowOutput> UnorderedSetDataflowGraph::query_outputs(
    DataflowOutputQuery const &q) const {
  return filter(this->outputs, [&](DataflowOutput const &o) {
    return includes(q.nodes, o.node) && includes(q.output_idxs, o.idx);
  });
}

std::unordered_set<DataflowGraphInput>
    UnorderedSetDataflowGraph::get_inputs() const {
  return this->graph_inputs;
}

void UnorderedSetDataflowGraph::add_node_unsafe(
    Node const &node,
    std::vector<DataflowOutput> const &inputs,
    std::vector<DataflowOutput> const &outputs) {
  std::vector<OpenDataflowValue> open_inputs = transform(
      inputs, [](DataflowOutput const &o) { return OpenDataflowValue{o}; });
  this->add_node_unsafe(node, open_inputs, outputs);
}

void UnorderedSetDataflowGraph::add_node_unsafe(
    Node const &node,
    std::vector<OpenDataflowValue> const &inputs,
    std::vector<DataflowOutput> const &outputs) {
  assert(!contains(this->nodes, node));
  assert(are_disjoint(this->outputs, without_order(outputs)));

  this->nodes.insert(node);

  for (auto const &[input_idx, input_src] : enumerate_vector(inputs)) {
    this->edges.insert(open_dataflow_edge_from_src_and_dst(
        input_src, DataflowInput{node, input_idx}));
  }

  extend(this->outputs, outputs);
}

void UnorderedSetDataflowGraph::inplace_materialize_from(
    DataflowGraphView const &view) {
  std::unordered_set<Node> nodes = get_nodes(view);
  std::unordered_set<DataflowEdge> edges = get_edges(view);
  std::unordered_set<DataflowOutput> outputs = get_all_dataflow_outputs(view);

  this->nodes = nodes;
  this->edges = transform(
      edges, [](DataflowEdge const &e) { return OpenDataflowEdge{e}; });
  this->outputs = outputs;
}

UnorderedSetDataflowGraph *UnorderedSetDataflowGraph::clone() const {
  return new UnorderedSetDataflowGraph{
      this->node_source,
      this->graph_input_source,
      this->nodes,
      this->edges,
      this->outputs,
      this->graph_inputs,
  };
}

} // namespace FlexFlow

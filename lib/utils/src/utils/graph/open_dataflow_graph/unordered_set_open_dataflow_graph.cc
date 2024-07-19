#include "utils/graph/open_dataflow_graph/unordered_set_open_dataflow_graph.h"

namespace FlexFlow {

UnorderedSetOpenDataflowGraph::UnorderedSetOpenDataflowGraph() {}

UnorderedSetOpenDataflowGraph::UnorderedSetOpenDataflowGraph(
    NodeSource const &node_source,
    DataflowGraphInputSource const &input_source,
    std::unordered_set<Node> const &nodes,
    std::unordered_set<DataflowEdge> const &standard_edges,
    std::unordered_set<DataflowInputEdge> const &input_edges,
    std::unordered_set<DataflowOutput> const &outputs,
    std::unordered_set<DataflowGraphInput> const &graph_inputs)
    : node_source(node_source), input_source(input_source), nodes(nodes),
      standard_edges(standard_edges), input_edges(input_edges),
      outputs(outputs), graph_inputs(graph_inputs) {}

NodeAddedResult UnorderedSetOpenDataflowGraph::add_node(
    std::vector<OpenDataflowValue> const &inputs, int num_outputs) {
  NOT_IMPLEMENTED();
}

std::unordered_set<Node>
    UnorderedSetOpenDataflowGraph::query_nodes(NodeQuery const &q) const {
  return apply_query(q.nodes, this->nodes);
}

std::unordered_set<OpenDataflowEdge> UnorderedSetOpenDataflowGraph::query_edges(
    OpenDataflowEdgeQuery const &q) const {
  std::unordered_set<DataflowEdge> standard_edges =
      filter(this->standard_edges, [&](DataflowEdge const &e) {
        return includes(q.standard_edge_query.src_nodes, e.src.node) &&
               includes(q.standard_edge_query.dst_nodes, e.dst.node) &&
               includes(q.standard_edge_query.src_idxs, e.src.idx) &&
               includes(q.standard_edge_query.dst_idxs, e.dst.idx);
      });
  std::unordered_set<DataflowInputEdge> input_edges =
      filter(this->input_edges, [&](DataflowInputEdge const &e) {
        return includes(q.input_edge_query.srcs, e.src) &&
               includes(q.input_edge_query.dst_nodes, e.dst.node) &&
               includes(q.input_edge_query.dst_idxs, e.dst.idx);
      });
  return set_union(
      transform(standard_edges,
                [](DataflowEdge const &e) { return OpenDataflowEdge{e}; }),
      transform(input_edges, [](DataflowInputEdge const &e) {
        return OpenDataflowEdge{e};
      }));
}

std::unordered_set<DataflowOutput> UnorderedSetOpenDataflowGraph::query_outputs(
    DataflowOutputQuery const &q) const {
  return filter(this->outputs, [&](DataflowOutput const &o) {
    return includes(q.nodes, o.node) && includes(q.output_idxs, o.idx);
  });
}

std::unordered_set<DataflowGraphInput>
    UnorderedSetOpenDataflowGraph::get_inputs() const {
  return this->graph_inputs;
}

DataflowGraphInput UnorderedSetOpenDataflowGraph::add_input() {
  DataflowGraphInput new_input = this->input_source.new_dataflow_graph_input();
  this->graph_inputs.insert(new_input);
  return new_input;
}

UnorderedSetOpenDataflowGraph *UnorderedSetOpenDataflowGraph::clone() const {
  return new UnorderedSetOpenDataflowGraph{
      this->node_source,
      this->input_source,
      this->nodes,
      this->standard_edges,
      this->input_edges,
      this->outputs,
      this->graph_inputs,
  };
}

} // namespace FlexFlow

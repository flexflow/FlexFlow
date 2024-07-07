#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

UnorderedSetDataflowGraph::UnorderedSetDataflowGraph() {}
UnorderedSetDataflowGraph::UnorderedSetDataflowGraph(NodeSource const &node_source, 
                                                     std::unordered_set<Node> const &nodes,
                                                     std::unordered_set<DataflowEdge> const &edges,
                                                     std::unordered_set<DataflowOutput> const &outputs)
  : node_source(node_source), nodes(nodes), edges(edges), outputs(outputs)
{}

NodeAddedResult UnorderedSetDataflowGraph::add_node(std::vector<DataflowOutput> const &inputs,
                                                    int num_outputs) {
  Node new_node = this->node_source.new_node();

  std::vector<DataflowOutput> new_outputs = transform(count(num_outputs),
                                                      [&](int output_idx) { return DataflowOutput{new_node, output_idx}; });

  this->add_node_unsafe(new_node, inputs, new_outputs);

  return NodeAddedResult{new_node, new_outputs};
}

std::unordered_set<Node> UnorderedSetDataflowGraph::query_nodes(NodeQuery const &q) const {
  return apply_query(q.nodes, this->nodes);
}

std::unordered_set<DataflowEdge> UnorderedSetDataflowGraph::query_edges(DataflowEdgeQuery const &q) const {
  return filter(this->edges, [&](DataflowEdge const &e) { 
    return includes(q.src_nodes, e.src.node) 
      && includes(q.dst_nodes, e.dst.node)
      && includes(q.src_idxs, e.src.idx)
      && includes(q.dst_idxs, e.dst.idx);
  });
}

std::unordered_set<DataflowOutput> UnorderedSetDataflowGraph::query_outputs(DataflowOutputQuery const &q) const {
  return filter(this->outputs, [&](DataflowOutput const &o) {
    return includes(q.nodes, o.node)
      && includes(q.output_idxs, o.idx);
  });
}

void UnorderedSetDataflowGraph::add_node_unsafe(Node const &node,
                     std::vector<DataflowOutput> const &inputs,
                     std::vector<DataflowOutput> const &outputs) {
  assert (!contains(this->nodes, node));
  assert (are_disjoint(this->outputs, without_order(outputs)));

  this->nodes.insert(node);

  for (auto const &[input_idx, input_src] : enumerate_vector(inputs)) {
    this->edges.insert(DataflowEdge{input_src, DataflowInput{node, input_idx}});
  }

  extend(this->outputs, outputs);
}

void UnorderedSetDataflowGraph::inplace_materialize_from(DataflowGraphView const &view) {
  std::unordered_set<Node> nodes = get_nodes(view);
  std::unordered_set<DataflowEdge> edges = get_edges(view);
  std::unordered_set<DataflowOutput> outputs = get_all_dataflow_outputs(view);

  this->nodes = nodes;
  this->edges = edges;
  this->outputs = outputs;
}

UnorderedSetDataflowGraph *UnorderedSetDataflowGraph::clone() const {
  return new UnorderedSetDataflowGraph{
    this->node_source, 
    this->nodes,
    this->edges,
    this->outputs,
  };
}


} // namespace FlexFlow

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_UNORDERED_SET_LABELLED_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_UNORDERED_SET_LABELLED_DATAFLOW_GRAPH_H

#include "utils/containers/enumerate_vector.h"
#include "utils/graph/dataflow_graph/dataflow_edge_query.h"
#include "utils/graph/dataflow_graph/dataflow_output_query.h"
#include "utils/graph/labelled_open_dataflow_graph/i_labelled_open_dataflow_graph.h"
#include "utils/graph/node/node_source.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.h"
#include "utils/containers/zip_vectors.h"
#include "utils/containers/without_nullopts.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
struct UnorderedSetLabelledOpenDataflowGraph final : public ILabelledOpenDataflowGraph<NodeLabel, ValueLabel> {
public:
  UnorderedSetLabelledOpenDataflowGraph() = default;

  NodeAddedResult add_node(NodeLabel const &node_label,
                           std::vector<OpenDataflowValue> const &inputs,
                           std::vector<ValueLabel> const &output_labels) override {
    Node new_node = this->node_source.new_node();
    this->nodes.insert({new_node, node_label});
    
    for (auto const &[input_idx, input] : enumerate_vector(inputs)) {
      this->edges.insert(open_dataflow_edge_from_src_and_dst(input, DataflowInput{new_node, input_idx}));
    }

    std::vector<DataflowOutput> new_outputs = transform(
      count(output_labels.size()),
      [&](int output_idx) { return DataflowOutput{new_node, output_idx}; });

    for (auto const &[output, output_label] : zip(new_outputs, output_labels)) {
      this->values.insert({OpenDataflowValue{output}, output_label});
    }

    return NodeAddedResult {
      new_node,
      new_outputs,
    };
  }
  
  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return filter(keys(this->nodes),
                  [&](Node const &n) { return includes(q.nodes, n); });
  }

  std::unordered_set<OpenDataflowEdge> query_edges(OpenDataflowEdgeQuery const &q) const override {
    return filter(this->edges,
                  [&](OpenDataflowEdge const &e) {
                    return open_dataflow_edge_query_includes(q, e);
                  });
  }

  std::unordered_set<DataflowOutput> query_outputs(DataflowOutputQuery const &q) const override {
    return without_nullopts(transform(keys(this->values), 
                                      [&](OpenDataflowValue const &v) -> std::optional<DataflowOutput> {
                                        DataflowOutput o = v.get<DataflowOutput>(); 
                                        if (dataflow_output_query_includes_dataflow_output(q, o)) {
                                          return o;
                                        } else {
                                         return std::nullopt;
                                        }
                                      }));
  }

  std::vector<DataflowGraphInput> get_inputs() const override {
    return this->inputs;
  }

  NodeLabel const &at(Node const &n) const override {
    return this->nodes.at(n);
  }

  ValueLabel const &at(OpenDataflowValue const &v) const override {
    return this->values.at(v);
  }

  UnorderedSetLabelledOpenDataflowGraph *clone() const override {
    return new UnorderedSetLabelledOpenDataflowGraph{
      this->node_source,
      this->inputs,
      this->nodes,
      this->edges,
      this->values,
    };
  }
private:
  UnorderedSetLabelledOpenDataflowGraph(NodeSource const &node_source,
                                        std::vector<DataflowGraphInput> const &inputs,
                                        std::unordered_map<Node, NodeLabel> const &nodes,
                                        std::unordered_set<OpenDataflowEdge> const &edges,
                                        std::unordered_map<OpenDataflowValue, ValueLabel> const &values)
    : node_source(node_source),
      inputs(inputs),
      nodes(nodes),
      edges(edges),
      values(values)
  {}
      
private:
  NodeSource node_source;
  std::vector<DataflowGraphInput> inputs;
  std::unordered_map<Node, NodeLabel> nodes;
  std::unordered_set<OpenDataflowEdge> edges;
  std::unordered_map<OpenDataflowValue, ValueLabel> values;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(UnorderedSetLabelledOpenDataflowGraph<int, int>);

} // namespace FlexFlow

#endif

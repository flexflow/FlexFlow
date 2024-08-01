#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_UNORDERED_SET_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_UNORDERED_SET_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/i_dataflow_graph.h"
#include "utils/graph/node/node_source.h"
#include "utils/graph/open_dataflow_graph/dataflow_graph_input_source.h"
#include "utils/graph/open_dataflow_graph/i_open_dataflow_graph.h"

namespace FlexFlow {

struct UnorderedSetDataflowGraph final : virtual public IDataflowGraph,
                                         virtual public IOpenDataflowGraph {
public:
  UnorderedSetDataflowGraph();

  NodeAddedResult add_node(std::vector<DataflowOutput> const &inputs,
                           int num_outputs) override;
  NodeAddedResult add_node(std::vector<OpenDataflowValue> const &inputs,
                           int num_outputs) override;
  DataflowGraphInput add_input() override;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
  std::unordered_set<OpenDataflowEdge>
      query_edges(OpenDataflowEdgeQuery const &) const override;
  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &) const override;
  std::unordered_set<DataflowGraphInput> get_inputs() const override;

  void add_node_unsafe(Node const &node,
                       std::vector<DataflowOutput> const &inputs,
                       std::vector<DataflowOutput> const &outputs) override;

  void inplace_materialize_from(DataflowGraphView const &view) override;

  UnorderedSetDataflowGraph *clone() const override;

private:
  void add_node_unsafe(Node const &node,
                       std::vector<OpenDataflowValue> const &inputs,
                       std::vector<DataflowOutput> const &outputs);

  UnorderedSetDataflowGraph(
      NodeSource const &node_source,
      DataflowGraphInputSource const &graph_input_source,
      std::unordered_set<Node> const &nodes,
      std::unordered_set<OpenDataflowEdge> const &edges,
      std::unordered_set<DataflowOutput> const &outputs,
      std::unordered_set<DataflowGraphInput> const &graph_inputs);

private:
  NodeSource node_source;
  DataflowGraphInputSource graph_input_source;
  std::unordered_set<Node> nodes;
  std::unordered_set<OpenDataflowEdge> edges;
  std::unordered_set<DataflowOutput> outputs;
  std::unordered_set<DataflowGraphInput> graph_inputs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(UnorderedSetDataflowGraph);

} // namespace FlexFlow

#endif

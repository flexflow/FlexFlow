#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_UNORDERED_SET_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_UNORDERED_SET_OPEN_DATAFLOW_GRAPH_H

#include "utils/graph/node/node_source.h"
#include "utils/graph/open_dataflow_graph/dataflow_graph_input_source.h"
#include "utils/graph/open_dataflow_graph/i_open_dataflow_graph.h"

namespace FlexFlow {

struct UnorderedSetOpenDataflowGraph : public IOpenDataflowGraph {
public:
  UnorderedSetOpenDataflowGraph();

  NodeAddedResult add_node(std::vector<OpenDataflowValue> const &inputs,
                           int num_outputs) override;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
  std::unordered_set<OpenDataflowEdge>
      query_edges(OpenDataflowEdgeQuery const &) const override;
  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &) const override;
  std::unordered_set<DataflowGraphInput> get_inputs() const override;

  DataflowGraphInput add_input() override;
  UnorderedSetOpenDataflowGraph *clone() const override;

private:
  UnorderedSetOpenDataflowGraph(
      NodeSource const &node_source,
      DataflowGraphInputSource const &input_source,
      std::unordered_set<Node> const &nodes,
      std::unordered_set<DataflowEdge> const &standard_edges,
      std::unordered_set<DataflowInputEdge> const &input_edges,
      std::unordered_set<DataflowOutput> const &outputs,
      std::unordered_set<DataflowGraphInput> const &graph_inputs);

private:
  NodeSource node_source;
  DataflowGraphInputSource input_source;
  std::unordered_set<Node> nodes;
  std::unordered_set<DataflowEdge> standard_edges;
  std::unordered_set<DataflowInputEdge> input_edges;
  std::unordered_set<DataflowOutput> outputs;
  std::unordered_set<DataflowGraphInput> graph_inputs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(UnorderedSetOpenDataflowGraph);

} // namespace FlexFlow

#endif

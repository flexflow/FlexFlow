#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_UNORDERED_SET_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_UNORDERED_SET_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/i_dataflow_graph.h"
#include "utils/graph/node/node_source.h"

namespace FlexFlow {

struct UnorderedSetDataflowGraph final : public IDataflowGraph {
public:
  UnorderedSetDataflowGraph();

  NodeAddedResult add_node(std::vector<DataflowOutput> const &inputs,
                           int num_outputs) override;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
  std::unordered_set<DataflowEdge> query_edges(DataflowEdgeQuery const &) const override;
  std::unordered_set<DataflowOutput> query_outputs(DataflowOutputQuery const &) const override;

  void add_node_unsafe(Node const &node,
                       std::vector<DataflowOutput> const &inputs,
                       std::vector<DataflowOutput> const &outputs) override;

  void inplace_materialize_from(DataflowGraphView const &view) override;

  UnorderedSetDataflowGraph *clone() const override;
private:
  UnorderedSetDataflowGraph(NodeSource const &node_source,
                            std::unordered_set<Node> const &nodes,
                            std::unordered_set<DataflowEdge> const &edges,
                            std::unordered_set<DataflowOutput> const &outputs);

private:
  NodeSource node_source;
  std::unordered_set<Node> nodes;
  std::unordered_set<DataflowEdge> edges;
  std::unordered_set<DataflowOutput> outputs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(UnorderedSetDataflowGraph);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPERATOR_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPERATOR_GRAPH_H

#include "utils/graph.h"
#include "pcg/operator_graph/operator_graph_output.dtg.h"
#include "pcg/operator_graph/operator_graph_input.dtg.h"

namespace FlexFlow {

struct OperatorGraphOutputQuery { };
struct OperatorGraphEdge { };

Node get_src_node(OperatorGraphEdge const &);
Node get_dst_node(OperatorGraphEdge const &);
int get_src_idx(OperatorGraphEdge const &);
int get_dst_idx(OperatorGraphEdge const &);

struct OperatorGraphEdgeQuery;

struct OperatorGraphView {
public:
  using Edge = OperatorGraphEdge;
  using EdgeQuery = OperatorGraphEdgeQuery;

  OperatorGraphView(OperatorGraphView const &);
  OperatorGraphView &operator=(OperatorGraphView const &);

  OperatorGraphView(OperatorGraphView &&);
  OperatorGraphView &&operator=(OperatorGraphView &&);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<OperatorGraphOutput> query_outputs(OperatorGraphOutputQuery const &) const;
  std::unordered_set<OperatorGraphEdge> query_edges(OperatorGraphEdgeQuery const &) const;

  struct Impl;
  std::unique_ptr<Impl> impl;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(OperatorGraphView);

std::unordered_set<OperatorGraphOutput> get_outputs(OperatorGraphView const &);
std::vector<OperatorGraphOutput> get_outputs(OperatorGraphView const &, Node const &);
std::unordered_set<OperatorGraphInput> get_uses(OperatorGraphView const &, OperatorGraphOutput const &);

struct OperatorGraph {
public:
  OperatorGraph();
  OperatorGraph(OperatorGraph const &) = default;
  OperatorGraph &operator=(OperatorGraph const &) = default;

  Node add_node(std::vector<OperatorGraphOutput> const &inputs, int num_outputs);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

struct value_t;

template <typename NodeLabel, typename OutputLabel>
struct LabelledOperatorGraphView : virtual OperatorGraphView {
   NodeLabel const &at(Node const &) const;
   OutputLabel const &at(OperatorGraphOutput const &) const;
};

template <typename NodeLabel, typename OutputLabel>
struct LabelledOperatorGraph : virtual LabelledOperatorGraphView<NodeLabel, OutputLabel> {
  Node add_node(NodeLabel const &, std::vector<OperatorGraphOutput> const &inputs, std::vector<OutputLabel> const &output_labels);
};

} // namespace FlexFlow

#endif

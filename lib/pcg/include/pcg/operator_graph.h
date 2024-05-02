#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPERATOR_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_OPERATOR_GRAPH_H

#include "utils/graph.h"

namespace FlexFlow {

struct OperatorGraphOutput { };
struct OperatorGraphInput { };
struct OperatorGraphOutputQuery { };
struct OperatorGraphEdge { };

Node get_node(OperatorGraphOutput const &);
int get_idx(OperatorGraphOutput const &);

Node get_node(OperatorGraphInput const &);
int get_idx(OperatorGraphInput const &);

Node get_src_node(OperatorGraphEdge const &);
Node get_dst_node(OperatorGraphEdge const &);
int get_src_idx(OperatorGraphEdge const &);
int get_dst_idx(OperatorGraphEdge const &);

struct OperatorGraphEdgeQuery;

struct OperatorGraphView : virtual MultiDiGraphView {
public:
  using Edge = OperatorGraphEdge;
  using EdgeQuery = OperatorGraphEdgeQuery;

  OperatorGraphView(OperatorGraphView const &) = default;
  OperatorGraphView &operator=(OperatorGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<OperatorGraphOutput> query_outputs(OperatorGraphOutputQuery const &) const;
  std::unordered_set<OperatorGraphEdge> query_edges(OperatorGraphEdgeQuery const &) const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(MultiDiGraphView);

std::vector<OperatorGraphOutput> get_outputs(OperatorGraphView const &, Node const &);
std::unordered_set<OperatorGraphInput> get_uses(OperatorGraphView const &, OperatorGraphOutput const &);

struct OperatorGraph : virtual OperatorGraphView {
public:
  OperatorGraph() = delete;
  OperatorGraph(OperatorGraph const &) = default;
  OperatorGraph &operator=(OperatorGraph const &) = default;

  Node add_node(std::vector<OperatorGraphOutput> const &inputs, int num_outputs);
};

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

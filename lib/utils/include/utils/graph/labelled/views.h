#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_VIEWS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_VIEWS_H

#include "node_labelled_interfaces.h"
#include "standard_labelled_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel>
struct NodeLabelledMultiDiSubgraphView
    : public INodeLabelledMultiDiGraphView<NodeLabel> {};

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiSubgraphView
    : public ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
public:
  LabelledMultiDiSubgraphView() = delete;
  template <typename InputLabel, typename OutputLabel>
  explicit LabelledMultiDiSubgraphView(
      ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> const &,
      std::unordered_set<Node> const &);
};

template <typename NodeLabel, typename OutputLabel>
struct ViewMultiDiGraphAsOutputLabelled
    : public IOutputLabelledMultiDiGraphView<NodeLabel, OutputLabel> {
public:
  ViewMultiDiGraphAsOutputLabelled() = delete;
  explicit ViewMultiDiGraphAsOutputLabelled(
      MultiDiGraphView const &g,
      std::function<NodeLabel(Node const &)> const &node_label,
      std::function<OutputLabel(MultiDiOutput const &)> const &output_label)
      : g(g), node_label(node_label), output_label(output_label) {}

  virtual std::unordered_set<Node>
      query_nodes(NodeQuery const &q) const override {
    return g.query_nodes(q);
  }

  virtual std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &q) const override {
    return g.query_edges(q);
  }

  virtual NodeLabel const &at(Node const &n) const override {
    return node_label(n);
  }

  virtual OutputLabel &at(MultiDiOutput const &o) override {
    return output_label(o);
  }

private:
  MultiDiGraphView g;
  std::function<NodeLabel(Node const &)> node_label;
  std::function<OutputLabel(MultiDiOutput const &)> output_label;
};

CHECK_NOT_ABSTRACT(ViewMultiDiGraphAsOutputLabelled<test_types::hash_cmp,
                                                    test_types::hash_cmp>);

} // namespace FlexFlow

#endif

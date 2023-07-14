#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_VIEWS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_VIEWS_H

#include "node_labelled_interfaces.h"
#include "standard_labelled_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel>
struct NodeLabelledMultiDiSubgraphView : public INodeLabelledMultiDiGraphView<NodeLabel> {
};

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiSubgraphView : public ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
public:
  LabelledMultiDiSubgraphView() = delete;
  template <typename InputLabel, typename OutputLabel>
  explicit LabelledMultiDiSubgraphView(
      ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> const &,
      std::unordered_set<Node> const &);
};

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel>
struct LabelledUpwardMultiDiSubgraphView {
public:
  LabelledUpwardMultiDiSubgraphView() = delete;
  template <typename OutputLabel>
  explicit LabelledUpwardMultiDiSubgraphView(
      ILabelledOpenMultiDiGraphView<NodeLabel,
                                    EdgeLabel,
                                    InputLabel,
                                    OutputLabel> const &,
      std::unordered_set<Node> const &);
};

template <typename NodeLabel,
          typename EdgeLabel,
          typename OutputLabel = EdgeLabel>
struct LabelledDownwardMultiDiSubgraphView {
public:
  LabelledDownwardMultiDiSubgraphView() = delete;
  template <typename InputLabel>
  explicit LabelledDownwardMultiDiSubgraphView(
      ILabelledOpenMultiDiGraphView<NodeLabel,
                                    EdgeLabel,
                                    InputLabel,
                                    OutputLabel> const &,
      std::unordered_set<Node> const &);
};

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct LabelledOpenMultiDiSubgraphView
    : public ILabelledOpenMultiDiGraphView<NodeLabel,
                                           EdgeLabel,
                                           InputLabel,
                                           OutputLabel> {
public:
  LabelledOpenMultiDiSubgraphView() = delete;
  explicit LabelledOpenMultiDiSubgraphView(
      ILabelledOpenMultiDiGraphView<NodeLabel,
                                    EdgeLabel,
                                    InputLabel,
                                    OutputLabel> const &,
      std::unordered_set<Node> const &);

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  virtual InputLabel const &at(InputMultiDiEdge const &e) const override;
  virtual OutputLabel const &at(OutputMultiDiEdge const &e) const override;
  virtual EdgeLabel const &at(MultiDiEdge const &e) const override;
};

}

#endif

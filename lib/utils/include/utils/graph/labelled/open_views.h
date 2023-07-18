#ifndef _FLEXFLOW__UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_VIEWS_H
#define _FLEXFLOW__UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_VIEWS_H

#include "labelled_downward_open.h"
#include "labelled_downward_open_interfaces.h"
#include "labelled_open.h"
#include "labelled_open_interfaces.h"
#include "labelled_upward_open.h"
#include "labelled_upward_open_interfaces.h"
#include "standard_labelled.h"

namespace FlexFlow {

enum class InputSettings { INCLUDE, EXCLUDE };
enum class OutputSettings { INCLUDE, EXCLUDE };

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

template <InputSettings INPUT_SETTINGS,
          OutputSettings OUTPUT_SETTINGS,
          typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
struct LabelledOpenMultiDiSubgraphView;

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
struct LabelledOpenMultiDiSubgraphView<InputSettings::INCLUDE,
                                       OutputSettings::INCLUDE,
                                       NodeLabel,
                                       EdgeLabel,
                                       InputLabel,
                                       OutputLabel>
    : public ILabelledOpenMultiDiGraphView<NodeLabel,
                                           EdgeLabel,
                                           InputLabel,
                                           OutputLabel> {
public:
  using ResultType = LabelledOpenMultiDiGraphView<NodeLabel,
                                                  EdgeLabel,
                                                  InputLabel,
                                                  OutputLabel>;

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

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
struct LabelledOpenMultiDiSubgraphView<InputSettings::INCLUDE,
                                       OutputSettings::EXCLUDE,
                                       NodeLabel,
                                       EdgeLabel,
                                       InputLabel,
                                       OutputLabel>
    : public ILabelledUpwardOpenMultiDiGraphView<NodeLabel,
                                                 EdgeLabel,
                                                 InputLabel> {
  using ResultType =
      LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>;
};

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
struct LabelledOpenMultiDiSubgraphView<InputSettings::EXCLUDE,
                                       OutputSettings::INCLUDE,
                                       NodeLabel,
                                       EdgeLabel,
                                       InputLabel,
                                       OutputLabel>
    : public ILabelledDownwardOpenMultiDiGraphView<NodeLabel,
                                                   EdgeLabel,
                                                   OutputLabel> {
  using ResultType =
      LabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>;
};

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
struct LabelledOpenMultiDiSubgraphView<InputSettings::EXCLUDE,
                                       OutputSettings::EXCLUDE,
                                       NodeLabel,
                                       EdgeLabel,
                                       InputLabel,
                                       OutputLabel>
    : public ILabelledMultiDiGraphView<NodeLabel, EdgeLabel> {
  using ResultType = LabelledMultiDiGraphView<NodeLabel, EdgeLabel>;
};

} // namespace FlexFlow

#endif

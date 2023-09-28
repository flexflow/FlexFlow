#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_ALGORITHMS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_ALGORITHMS_H

#include "open_views.h"
#include "utils/graph/labelled/labelled_upward_open.h"

namespace FlexFlow {

enum class OpenType { OPEN, UPWARD, DOWNWARD, CLOSED };

template <OpenType OPEN_TYPE>
struct get_input_settings;

template <>
struct get_input_settings<OpenType::OPEN>
    : std::integral_constant<InputSettings, InputSettings::INCLUDE> {};

template <>
struct get_input_settings<OpenType::UPWARD>
    : std::integral_constant<InputSettings, InputSettings::INCLUDE> {};

template <>
struct get_input_settings<OpenType::DOWNWARD>
    : std::integral_constant<InputSettings, InputSettings::EXCLUDE> {};

template <>
struct get_input_settings<OpenType::CLOSED>
    : std::integral_constant<InputSettings, InputSettings::EXCLUDE> {};

template <OpenType OPEN_TYPE>
struct get_output_settings;

template <>
struct get_output_settings<OpenType::OPEN>
    : std::integral_constant<OutputSettings, OutputSettings::INCLUDE> {};

template <>
struct get_output_settings<OpenType::UPWARD>
    : std::integral_constant<OutputSettings, OutputSettings::EXCLUDE> {};

template <>
struct get_output_settings<OpenType::DOWNWARD>
    : std::integral_constant<OutputSettings, OutputSettings::INCLUDE> {};

template <>
struct get_output_settings<OpenType::CLOSED>
    : std::integral_constant<OutputSettings, OutputSettings::EXCLUDE> {};

template <InputSettings INPUT_SETTINGS,
          OutputSettings OUTPUT_SETTINGS,
          typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel,
          typename ResultType =
              typename LabelledOpenMultiDiSubgraphView<INPUT_SETTINGS,
                                                       OUTPUT_SETTINGS,
                                                       NodeLabel,
                                                       EdgeLabel,
                                                       InputLabel,
                                                       OutputLabel>::ResultType>
ResultType get_subgraph(LabelledOpenMultiDiGraphView<NodeLabel,
                                                     EdgeLabel,
                                                     InputLabel,
                                                     OutputLabel> const &g,
                        std::unordered_set<Node> const &nodes) {
  return ResultType::template create<
      LabelledOpenMultiDiSubgraphView<INPUT_SETTINGS,
                                      OUTPUT_SETTINGS,
                                      NodeLabel,
                                      EdgeLabel,
                                      InputLabel,
                                      OutputLabel>>(g, nodes);
}

template <InputSettings INPUT_SETTINGS,
          OutputSettings OUTPUT_SETTINGS,
          typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel,
          typename ResultType =
              typename LabelledOpenMultiDiSubgraphView<INPUT_SETTINGS,
                                                       OUTPUT_SETTINGS,
                                                       NodeLabel,
                                                       EdgeLabel,
                                                       InputLabel,
                                                       OutputLabel>::ResultType>
ResultType get_subgraph(LabelledOpenMultiDiGraph<NodeLabel,
                                                 EdgeLabel,
                                                 InputLabel,
                                                 OutputLabel> const &g,
                        std::unordered_set<Node> const &nodes) {
  return get_subgraph<INPUT_SETTINGS, OUTPUT_SETTINGS>(as_view(g), nodes);
}

template <
    OpenType OPEN_TYPE,
    typename NodeLabel,
    typename EdgeLabel,
    typename InputLabel,
    typename OutputLabel,
    InputSettings INPUT_SETTINGS = get_input_settings<OPEN_TYPE>::value,
    OutputSettings OUTPUT_SETTINGS = get_output_settings<OPEN_TYPE>::value,
    typename ResultType =
        typename LabelledOpenMultiDiSubgraphView<INPUT_SETTINGS,
                                                 OUTPUT_SETTINGS,
                                                 NodeLabel,
                                                 EdgeLabel,
                                                 InputLabel,
                                                 OutputLabel>::ResultType>
ResultType get_subgraph(LabelledOpenMultiDiGraphView<NodeLabel,
                                                     EdgeLabel,
                                                     InputLabel,
                                                     OutputLabel> const &g,
                        std::unordered_set<Node> const &nodes) {
  return get_subgraph<INPUT_SETTINGS,
                      OUTPUT_SETTINGS,
                      NodeLabel,
                      EdgeLabel,
                      InputLabel,
                      OutputLabel>(g, nodes);
}

template <
    OpenType OPEN_TYPE,
    typename NodeLabel,
    typename EdgeLabel,
    typename InputLabel,
    typename OutputLabel,
    InputSettings INPUT_SETTINGS = get_input_settings<OPEN_TYPE>::value,
    OutputSettings OUTPUT_SETTINGS = get_output_settings<OPEN_TYPE>::value,
    typename ResultType =
        typename LabelledOpenMultiDiSubgraphView<INPUT_SETTINGS,
                                                 OUTPUT_SETTINGS,
                                                 NodeLabel,
                                                 EdgeLabel,
                                                 InputLabel,
                                                 OutputLabel>::ResultType>
ResultType get_subgraph(LabelledOpenMultiDiGraph<NodeLabel,
                                                 EdgeLabel,
                                                 InputLabel,
                                                 OutputLabel> const &g,
                        std::unordered_set<Node> const &nodes) {
  return get_subgraph<INPUT_SETTINGS,
                      OUTPUT_SETTINGS,
                      NodeLabel,
                      EdgeLabel,
                      InputLabel,
                      OutputLabel>(as_view(g), nodes);
}

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>
    as_upward_open(LabelledOpenMultiDiGraphView<NodeLabel,
                                                EdgeLabel,
                                                InputLabel,
                                                OutputLabel> const &g) {
  return LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::
      template create<ViewLabelledOpenMultiDiGraphAsUpwardOpen<NodeLabel,
                                                               EdgeLabel,
                                                               InputLabel,
                                                               OutputLabel>>(g);
}

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
LabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>
    as_downward_open(LabelledOpenMultiDiGraphView<NodeLabel,
                                                  EdgeLabel,
                                                  InputLabel,
                                                  OutputLabel> const &g) {
  return LabelledDownwardOpenMultiDiGraphView<NodeLabel,
                                              EdgeLabel,
                                              InputLabel>::
      template create<ViewLabelledOpenMultiDiGraphAsDownwardOpen<NodeLabel,
                                                                 EdgeLabel,
                                                                 InputLabel,
                                                                 OutputLabel>>(
          g);
}

} // namespace FlexFlow

#endif

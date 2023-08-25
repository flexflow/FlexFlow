#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_OUTPUT_LABELLED_OPEN

namespace FlexFlow {

template <typename NodeLabel,
          typename InputLabel,
          typename OutputLabel = InputLabel>
struct OutputLabelledOpenMultiDiGraph {
  OutputLabelledOpenMultiDiGraph() = delete;
  OutputLabelledOpenMultiDiGraph(OutputLabelledOpenMultiDiGraph const &) =
      default;
  OutputLabelledOpenMultiDiGraph &
      operator=(OutputLabelledOpenMultiDiGraph const &) = default;

  operator OpenMultiDiGraphView();
};

} // namespace FlexFlow

#endif

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

  operator OpenMultiDiGraphView() const {
    NOT_IMPLEMENTED();
  }

  Node add_node(NodeLabel const &) {
    NOT_IMPLEMENTED();
  }
  NodeLabel const &at(Node const &) const {
    NOT_IMPLEMENTED();
  }
  NodeLabel &at(Node const &) {
    NOT_IMPLEMENTED();
  }

  void add_edge(MultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }
  void add_edge(InputMultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }
  void add_edge(OutputMultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }

  InputLabel const &at(InputMultiDiEdge const &) const {
    NOT_IMPLEMENTED();
  }
  OutputLabel const &at(OutputMultiDiEdge const &) const {
    NOT_IMPLEMENTED();
  }

  InputLabel &at(InputMultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }
  OutputLabel &at(OutputMultiDiEdge const &) {
    NOT_IMPLEMENTED();
  }

  void add_output(MultiDiOutput const &, OutputLabel const &) {
    NOT_IMPLEMENTED();
  }
  OutputLabel const &at(MultiDiOutput const &) const {
    NOT_IMPLEMENTED();
  }
  OutputLabel &at(MultiDiOutput const &) {
    NOT_IMPLEMENTED();
  }

  OutputLabel const &at(OpenMultiDiEdge const &) const {
    NOT_IMPLEMENTED();
  }
};

} // namespace FlexFlow

#endif

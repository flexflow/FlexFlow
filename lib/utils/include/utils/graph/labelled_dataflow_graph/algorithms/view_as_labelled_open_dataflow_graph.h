#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_AS_OPEN_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_AS_OPEN_GRAPH_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
struct LabelledDataflowGraphAsOpenView final
    : public ILabelledOpenDataflowGraphView<NodeLabel, ValueLabel> {
public:
  LabelledDataflowGraphAsOpenView() = delete;
  LabelledDataflowGraphAsOpenView(
      LabelledDataflowGraphView<NodeLabel, ValueLabel> const &g)
      : g(g) {}

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->g.query_nodes(q);
  }

  std::unordered_set<OpenDataflowEdge>
      query_edges(OpenDataflowEdgeQuery const &q) const override {
    return transform(this->g.query_edges(q.standard_edge_query),
                     [](DataflowEdge const &e) { return OpenDataflowEdge{e}; });
  }

  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &q) const override {
    return this->g.query_outputs(q);
  }

  std::unordered_set<DataflowGraphInput> get_inputs() const override {
    return {};
  }

  NodeLabel const &at(Node const &n) const override {
    return this->g.at(n);
  }

  ValueLabel const &at(OpenDataflowValue const &v) const override {
    return this->g.at(v.get<DataflowOutput>());
  }

  LabelledDataflowGraphAsOpenView *clone() const override {
    return new LabelledDataflowGraphAsOpenView{this->g};
  }

private:
  LabelledDataflowGraphView<NodeLabel, ValueLabel> g;
};

template <typename NodeLabel, typename ValueLabel>
LabelledOpenDataflowGraphView<NodeLabel, ValueLabel>
    view_as_labelled_open_dataflow_graph(
        LabelledDataflowGraphView<NodeLabel, ValueLabel> const &g) {
  return LabelledOpenDataflowGraphView<NodeLabel, ValueLabel>::template create<
      LabelledDataflowGraphAsOpenView<NodeLabel, ValueLabel>>(g);
}

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_WITH_LABELLING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_WITH_LABELLING_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
struct OpenDataflowGraphLabellingWrapper final
    : public ILabelledOpenDataflowGraphView<NodeLabel, ValueLabel> {
public:
  OpenDataflowGraphLabellingWrapper() = delete;
  OpenDataflowGraphLabellingWrapper(
      OpenDataflowGraphView const &unlabelled,
      std::unordered_map<Node, NodeLabel> const &node_labels,
      std::unordered_map<OpenDataflowValue, ValueLabel> const &value_labels)
      : unlabelled(unlabelled), node_labels(node_labels),
        value_labels(value_labels) {}

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->unlabelled.query_nodes(q);
  }

  std::unordered_set<OpenDataflowEdge>
      query_edges(OpenDataflowEdgeQuery const &q) const override {
    return this->unlabelled.query_edges(q);
  }

  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &q) const override {
    return this->unlabelled.query_outputs(q);
  }

  std::unordered_set<DataflowGraphInput> get_inputs() const override {
    return this->unlabelled.get_inputs();
  }

  NodeLabel const &at(Node const &n) const override {
    return this->node_labels.at(n);
  }

  ValueLabel const &at(OpenDataflowValue const &v) const override {
    return this->value_labels.at(v);
  }

  OpenDataflowGraphLabellingWrapper *clone() const override {
    return new OpenDataflowGraphLabellingWrapper{
        this->unlabelled,
        this->node_labels,
        this->value_labels,
    };
  }

private:
  OpenDataflowGraphView unlabelled;
  std::unordered_map<Node, NodeLabel> node_labels;
  std::unordered_map<OpenDataflowValue, ValueLabel> value_labels;
};

template <typename NodeLabel, typename ValueLabel>
LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> with_labelling(
    OpenDataflowGraphView const &g,
    std::unordered_map<Node, NodeLabel> const &node_labels,
    std::unordered_map<OpenDataflowValue, ValueLabel> const &value_labels) {
  return LabelledOpenDataflowGraphView<NodeLabel, ValueLabel>::template create<
      OpenDataflowGraphLabellingWrapper<NodeLabel, ValueLabel>>(
      g, node_labels, value_labels);
}

} // namespace FlexFlow

#endif

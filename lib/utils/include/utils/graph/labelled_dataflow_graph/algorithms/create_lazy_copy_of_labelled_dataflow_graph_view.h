#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_CREATE_LAZY_COPY_OF_LABELLED_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_ALGORITHMS_CREATE_LAZY_COPY_OF_LABELLED_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/labelled_dataflow_graph/i_labelled_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph_view.h"
#include <functional>

namespace FlexFlow {

// NOTE(@lockshaw) This code is not tested and I don't necessarily trust it.
// Figuring out what to do with it is tracked in
// https://github.com/flexflow/FlexFlow/issues/1513

template <typename NodeLabel, typename ValueLabel>
struct LazyLabelledDataflowGraph final
    : public ILabelledDataflowGraph<NodeLabel, ValueLabel> {
public:
  LazyLabelledDataflowGraph() = delete;
  LazyLabelledDataflowGraph(
      LabelledDataflowGraphView<NodeLabel, ValueLabel> const &view,
      std::function<LabelledDataflowGraph<NodeLabel, ValueLabel>(
          LabelledDataflowGraphView<NodeLabel, ValueLabel> const &)> const
          &make_copy_func)
      : g(view), make_copy_func(make_copy_func) {}

  NodeAddedResult
      add_node(NodeLabel const &node_label,
               std::vector<DataflowOutput> const &inputs,
               std::vector<ValueLabel> const &output_labels) override {
    return this->get_mutable_graph().add_node(
        node_label, inputs, output_labels);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->get_view().query_nodes(q);
  }

  std::unordered_set<DataflowEdge>
      query_edges(DataflowEdgeQuery const &q) const override {
    return this->get_view().query_edges(q);
  }

  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &q) const override {
    return this->get_view().query_outputs(q);
  }

  NodeLabel at(Node const &n) const override {
    return this->get_view().at(n);
  }

  ValueLabel at(DataflowOutput const &v) const override {
    return this->get_view().at(v);
  }

  LazyLabelledDataflowGraph *clone() const override {
    return new LazyLabelledDataflowGraph(this->g, this->make_copy_func);
  }

  void inplace_materialize_from(
      LabelledDataflowGraphView<NodeLabel, ValueLabel> const &view) override {
    this->g = view;
  }

private:
  std::variant<LabelledDataflowGraphView<NodeLabel, ValueLabel>,
               LabelledDataflowGraph<NodeLabel, ValueLabel>>
      g;
  std::function<LabelledDataflowGraph<NodeLabel, ValueLabel>(
      LabelledDataflowGraphView<NodeLabel, ValueLabel> const &)>
      make_copy_func;

private:
  LazyLabelledDataflowGraph(decltype(g) const &g,
                            decltype(make_copy_func) const &make_copy_func)
      : g(g), make_copy_func(make_copy_func) {}

  LabelledDataflowGraphView<NodeLabel, ValueLabel> const &get_view() const {
    if (g.index() == 0) {
      return std::get<0>(this->g);
    } else {
      assert(g.index() == 1);
      return std::get<1>(this->g);
    }
  }

  LabelledDataflowGraph<NodeLabel, ValueLabel> &get_mutable_graph() {
    if (g.index() == 0) {
      this->g = this->make_copy_func(std::get<0>(g));
    }
    assert(g.index() == 1);

    return std::get<1>(g);
  }
};

template <typename T, typename NodeLabel, typename ValueLabel>
static typename std::enable_if<
    std::is_base_of<ILabelledDataflowGraph<NodeLabel, ValueLabel>, T>::value,
    LabelledDataflowGraph<NodeLabel, ValueLabel>>::type
    create_lazy_copy_of_labelled_dataflow_graph_view(
        LabelledDataflowGraphView<NodeLabel, ValueLabel> const &view) {
  std::function<LabelledDataflowGraph<NodeLabel, ValueLabel>(
      LabelledDataflowGraphView<NodeLabel, ValueLabel> const &)>
      make_copy_func = [](LabelledDataflowGraphView<NodeLabel, ValueLabel> const
                              &v) {
        return LabelledDataflowGraph<NodeLabel,
                                     ValueLabel>::template create_copy_of<T>(v);
      };
  return LabelledDataflowGraph<NodeLabel, ValueLabel>::template create<
      LazyLabelledDataflowGraph<NodeLabel, ValueLabel>>(view, make_copy_func);
}

} // namespace FlexFlow

#endif

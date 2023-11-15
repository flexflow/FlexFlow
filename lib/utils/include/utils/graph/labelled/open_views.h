#ifndef _FLEXFLOW__UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_VIEWS_H
#define _FLEXFLOW__UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_VIEWS_H

#include "output_labelled_open.h"
#include "standard_labelled.h"
#include "utils/exception.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/multidiedge.h"
#include "utils/graph/open_graph_interfaces.h"
#include "utils/graph/open_graphs.h"
#include "utils/type_traits.h"
#include "utils/visitable.h"

namespace FlexFlow {

template <typename SubgraphView, typename NodeLabel, typename EdgeLabel>
struct OutputLabelledOpenMultiDiSubgraphView
    : virtual IOutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> {

  OutputLabelledOpenMultiDiSubgraphView(
      OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> const &g,
      std::unordered_set<Node> const &nodes)
      : g(g), nodes(nodes) {}

  NodeLabel const &at(Node const &n) const override {
    return g.at(n);
  }

  EdgeLabel const &at(InputMultiDiEdge const &i) const override {
    return g.at(i);
  }

  EdgeLabel const &at(MultiDiOutput const &o) const override {
    return g.at(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return SubgraphView(g, nodes).query_nodes(q);
  }

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const override {
    return SubgraphView(g, nodes).query_edges(q);
  }

  OutputLabelledOpenMultiDiSubgraphView *clone() const override {
    return new OutputLabelledOpenMultiDiSubgraphView(g, nodes);
  }

private:
  OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> const &g;
  std::unordered_set<Node> const &nodes;
};

// CHECK_NOT_ABSTRACT(OutputLabelledOpenMultiDiSubgraphView);

template <typename NodeLabel, typename EdgeLabel>
struct ViewOutputLabelledAsOutputLabelledOpen
    : virtual IOutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel> {
  ViewOutputLabelledAsOutputLabelledOpen(
      OutputLabelledMultiDiGraphView<NodeLabel, EdgeLabel> const &g)
      : g(g) {}

  NodeLabel const &at(Node const &n) const override {
    return g.at(n);
  }

  EdgeLabel const &at(InputMultiDiEdge const &i) const override {
    assert(false);
  }

  EdgeLabel const &at(MultiDiOutput const &o) const override {
    return g.at(o);
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return g.query_nodes(q);
  }

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const override {
    return transform(g.query_edges(q.standard_edge_query),
                     [](MultiDiEdge const &e) { return OpenMultiDiEdge(e); });
  }

  ViewOutputLabelledAsOutputLabelledOpen *clone() const override {
    return new ViewOutputLabelledAsOutputLabelledOpen(g);
  }

private:
  OutputLabelledMultiDiGraphView<NodeLabel, EdgeLabel> const &g;
};

template <typename NodeLabel, typename EdgeLabel>
OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel>
    view_output_labelled_as_output_labelled_open(
        OutputLabelledMultiDiGraphView<NodeLabel, EdgeLabel> const &g) {
  return OutputLabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel>::
      template create<
          ViewOutputLabelledAsOutputLabelledOpen<NodeLabel, EdgeLabel>>(g);
}

} // namespace FlexFlow

#endif

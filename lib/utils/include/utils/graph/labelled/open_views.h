#ifndef _FLEXFLOW__UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_VIEWS_H
#define _FLEXFLOW__UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_VIEWS_H

#include "labelled_downward_open.h"
#include "labelled_downward_open_interfaces.h"
#include "labelled_open.h"
#include "labelled_open_interfaces.h"
#include "labelled_upward_open.h"
#include "labelled_upward_open_interfaces.h"
#include "standard_labelled.h"
#include "utils/exception.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/multidiedge.h"
#include "utils/graph/open_graph_interfaces.h"
#include "utils/graph/open_graphs.h"
#include "utils/type_traits.h"
#include "utils/visitable.h"

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

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
struct ViewLabelledOpenMultiDiGraphAsUpwardOpen
    : public ILabelledUpwardOpenMultiDiGraphView<NodeLabel,
                                                 EdgeLabel,
                                                 InputLabel> {
public:
  using InputType = LabelledOpenMultiDiGraphView<NodeLabel,
                                                 EdgeLabel,
                                                 InputLabel,
                                                 OutputLabel>;

  explicit ViewLabelledOpenMultiDiGraphAsUpwardOpen(InputType const &g)
      : g(g) {}

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->g.query_nodes(q);
  }

  std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &q) const override {
    return value_all(
        narrow<UpwardOpenMultiDiEdgeQuery>(this->g.query_edges(q)));
  }

  NodeLabel const &at(Node const &n) const override {
    return this->g.at(n);
  }
  InputLabel const &at(InputMultiDiEdge const &e) const override {
    return this->g.at(e);
  }
  EdgeLabel const &at(MultiDiEdge const &e) const override {
    return this->g.at(e);
  }

private:
  InputType g;
};
CHECK_NOT_ABSTRACT(
    ViewLabelledOpenMultiDiGraphAsUpwardOpen<test_types::hash_cmp,
                                             test_types::hash_cmp,
                                             test_types::hash_cmp,
                                             test_types::hash_cmp>);

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
struct ViewLabelledOpenMultiDiGraphAsDownwardOpen
    : public ILabelledDownwardOpenMultiDiGraphView<NodeLabel,
                                                   EdgeLabel,
                                                   OutputLabel> {
public:
  using InputType = LabelledOpenMultiDiGraphView<NodeLabel,
                                                 EdgeLabel,
                                                 InputLabel,
                                                 OutputLabel>;

  explicit ViewLabelledOpenMultiDiGraphAsDownwardOpen(InputType const &g)
      : g(g) {}

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->g.query_nodes(q);
  }

  std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &q) const override {
    return value_all(
        narrow<DownwardOpenMultiDiEdgeQuery>(this->g.query_edges(q)));
  }

  NodeLabel const &at(Node const &n) const override {
    return this->g.at(n);
  }
  OutputLabel const &at(OutputMultiDiEdge const &e) const override {
    return this->g.at(e);
  }
  EdgeLabel const &at(MultiDiEdge const &e) const override {
    return this->g.at(e);
  }

private:
  InputType g;
};
CHECK_NOT_ABSTRACT(
    ViewLabelledOpenMultiDiGraphAsUpwardOpen<test_types::hash_cmp,
                                             test_types::hash_cmp,
                                             test_types::hash_cmp,
                                             test_types::hash_cmp>);

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
      std::unordered_set<Node> const &) {
    NOT_IMPLEMENTED();
  }

  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &) const override {
    NOT_IMPLEMENTED();
  }
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override {
    NOT_IMPLEMENTED();
  }

  virtual NodeLabel const &at(Node const &n) const override {
    NOT_IMPLEMENTED();
  }
  virtual InputLabel const &at(InputMultiDiEdge const &e) const override {
    NOT_IMPLEMENTED();
  }
  virtual OutputLabel const &at(OutputMultiDiEdge const &e) const override {
    NOT_IMPLEMENTED();
  }
  virtual EdgeLabel const &at(MultiDiEdge const &e) const override {
    NOT_IMPLEMENTED();
  }
};
CHECK_NOT_ABSTRACT(LabelledOpenMultiDiSubgraphView<InputSettings::INCLUDE,
                                                   OutputSettings::INCLUDE,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp>);

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

  LabelledOpenMultiDiSubgraphView(
      LabelledOpenMultiDiGraphView<NodeLabel,
                                   EdgeLabel,
                                   InputLabel,
                                   OutputLabel> const &g,
      std::unordered_set<Node> const &nodes)
      : g(g), nodes(nodes) {}

  std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &q) const override {
    return static_cast<UpwardOpenMultiDiGraphView>(this->g).query_edges(q);
  }
  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return static_cast<UpwardOpenMultiDiGraphView>(this->g).query_nodes(q);

    NOT_IMPLEMENTED();
  }

  virtual NodeLabel const &at(Node const &n) const override {
    assert(contains(this->g, n));
    return g.at(n);
  }
  virtual InputLabel const &at(InputMultiDiEdge const &e) const override {
    assert(contains(this->g, e));
    return g.at(e);
  }
  virtual EdgeLabel const &at(MultiDiEdge const &e) const override {
    NOT_IMPLEMENTED();
  }

  std::unordered_set<Node> nodes;
  ResultType g;
};
CHECK_NOT_ABSTRACT(LabelledOpenMultiDiSubgraphView<InputSettings::INCLUDE,
                                                   OutputSettings::EXCLUDE,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp>);

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

  std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &) const override {
    NOT_IMPLEMENTED();
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override {
    NOT_IMPLEMENTED();
  }

  virtual NodeLabel const &at(Node const &n) const override {
    NOT_IMPLEMENTED();
  }

  virtual OutputLabel const &at(OutputMultiDiEdge const &e) const override {
    NOT_IMPLEMENTED();
  }

  virtual EdgeLabel const &at(MultiDiEdge const &e) const override {
    NOT_IMPLEMENTED();
  }
};
CHECK_NOT_ABSTRACT(LabelledOpenMultiDiSubgraphView<InputSettings::EXCLUDE,
                                                   OutputSettings::INCLUDE,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp>);

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

  std::unordered_set<MultiDiEdge>
      query_edges(MultiDiEdgeQuery const &) const override {
    NOT_IMPLEMENTED();
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &) const override {
    NOT_IMPLEMENTED();
  }

  virtual NodeLabel const &at(Node const &) const override {
    NOT_IMPLEMENTED();
  }

  virtual EdgeLabel const &at(MultiDiEdge const &e) const override {
    NOT_IMPLEMENTED();
  }
};
CHECK_NOT_ABSTRACT(LabelledOpenMultiDiSubgraphView<InputSettings::EXCLUDE,
                                                   OutputSettings::EXCLUDE,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp,
                                                   test_types::hash_cmp>);

} // namespace FlexFlow

#endif

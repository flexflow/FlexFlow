#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_DECL_H

#include "labelled_open_interfaces.h"
#include "node_labelled.h"
#include "utils/graph/open_graphs.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct LabelledOpenMultiDiGraphView {
private:
  using Interface = ILabelledOpenMultiDiGraphView<NodeLabel,
                                                  EdgeLabel,
                                                  InputLabel,
                                                  OutputLabel>;

public:
  LabelledOpenMultiDiGraphView() = delete;

  operator OpenMultiDiGraphView() const;
  // operator MultiDiGraphView() const;

  NodeLabel const &at(Node const &n) const;
  EdgeLabel const &at(MultiDiEdge const &e) const;
  InputLabel const &at(InputMultiDiEdge const &e) const;
  OutputLabel const &at(OutputMultiDiEdge const &e) const;

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledOpenMultiDiGraphView>::type
      create();

private:
  std::shared_ptr<Interface const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(
    LabelledOpenMultiDiGraphView<int, int, int, int>);

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = InputLabel>
struct LabelledOpenMultiDiGraph {
private:
  using Interface =
      ILabelledOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel, OutputLabel>;

public:
  LabelledOpenMultiDiGraph() = delete;
  LabelledOpenMultiDiGraph(LabelledOpenMultiDiGraph const &other) = default;
  LabelledOpenMultiDiGraph &
      operator=(LabelledOpenMultiDiGraph const &other) = default;

  operator LabelledOpenMultiDiGraphView<NodeLabel,
                                        EdgeLabel,
                                        InputLabel,
                                        OutputLabel>() const;

  operator OpenMultiDiGraphView() const;

  friend void swap(LabelledOpenMultiDiGraph &lhs,
                   LabelledOpenMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  Node add_node(NodeLabel const &l);
  NodeLabel &at(Node const &n);

  NodePort add_node_port();

  NodeLabel const &at(Node const &n) const;

  void add_node_unsafe(Node const &n, NodeLabel const &l);

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const;
  std::unordered_set<OpenMultiDiEdge>
      query_edges(OpenMultiDiEdgeQuery const &q) const;

  void add_edge(
      MultiDiEdge const &e); // We should allow adding edges without labels. For
                             // example, we may want to first construct a PCG
                             // and infer its tensor shapes later.
  void add_edge(InputMultiDiEdge const &e);
  void add_edge(OutputMultiDiEdge const &e);

  void add_label(MultiDiEdge const &e, EdgeLabel const &l);
  void add_label(InputMultiDiEdge const &e, EdgeLabel const &l);
  void add_label(OutputMultiDiEdge const &e, EdgeLabel const &l);

  void add_edge(MultiDiEdge const &e, EdgeLabel const &l);
  EdgeLabel &at(MultiDiEdge const &e);
  EdgeLabel const &at(MultiDiEdge const &e) const;

  void add_edge(InputMultiDiEdge const &e, InputLabel const &l);
  InputLabel &at(InputMultiDiEdge const &e);
  InputLabel const &at(InputMultiDiEdge const &e) const;

  void add_edge(OutputMultiDiEdge const &, OutputLabel const &);
  OutputLabel &at(OutputMultiDiEdge const &);
  OutputLabel const &at(OutputMultiDiEdge const &) const;

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledOpenMultiDiGraph>::type
      create();

private:
  LabelledOpenMultiDiGraph(cow_ptr_t<Interface> ptr);

private:
  cow_ptr_t<Interface> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(
    LabelledOpenMultiDiGraph<int, int, int, int>);

} // namespace FlexFlow

#endif

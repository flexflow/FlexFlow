#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_UPWARD_OPEN_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_UPWARD_OPEN_MULTIDIGRAPH_H

#include "utils/graph/upward_open_multidigraph/i_upward_open_multidigraph.h"
#include "utils/graph/upward_open_multidigraph/upward_open_multidigraph_view.h"

namespace FlexFlow {

struct UpwardOpenMultiDiGraph : virtual UpwardOpenMultiDiGraphView {
public:
  using Edge = UpwardOpenMultiDiEdge;
  using EdgeQuery = UpwardOpenMultiDiEdgeQuery;

  UpwardOpenMultiDiGraph() = delete;
  UpwardOpenMultiDiGraph(UpwardOpenMultiDiGraph const &) = default;
  UpwardOpenMultiDiGraph &operator=(UpwardOpenMultiDiGraph const &) = default;

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<
      std::is_base_of<IUpwardOpenMultiDiGraph, T>::value,
      UpwardOpenMultiDiGraph>::type
      create() {
    return UpwardOpenMultiDiGraph(make_cow_ptr<T>());
  }

private:
  using UpwardOpenMultiDiGraphView::UpwardOpenMultiDiGraphView;

  IUpwardOpenMultiDiGraph const &get_ptr() const;
  IUpwardOpenMultiDiGraph &get_ptr();
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UpwardOpenMultiDiGraph);


} // namespace FlexFlow

#endif

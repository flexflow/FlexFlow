#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_DOWNWARD_OPEN_MULTIDIGRAPH_H

#include "utils/graph/downward_open_multidigraph/downward_open_multidigraph_view.h"
#include "utils/graph/downward_open_multidigraph/i_downward_open_multidigraph.h"

namespace FlexFlow {

struct DownwardOpenMultiDiGraph : virtual DownwardOpenMultiDiGraphView {
public:
  using Edge = DownwardOpenMultiDiEdge;
  using EdgeQuery = DownwardOpenMultiDiEdgeQuery;

  DownwardOpenMultiDiGraph() = delete;
  DownwardOpenMultiDiGraph(DownwardOpenMultiDiGraph const &) = default;
  DownwardOpenMultiDiGraph &
      operator=(DownwardOpenMultiDiGraph const &) = default;

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<
      std::is_base_of<IDownwardOpenMultiDiGraph, T>::value,
      DownwardOpenMultiDiGraph>::type
      create() {
    return DownwardOpenMultiDiGraph(make_cow_ptr<T>());
  }

private:
  using DownwardOpenMultiDiGraphView::DownwardOpenMultiDiGraphView;

  IDownwardOpenMultiDiGraph &get_ptr();
  IDownwardOpenMultiDiGraph const &get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DownwardOpenMultiDiGraph);


} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIGRAPH_H

#include "utils/graph/multidigraph/i_multidigraph.h"
#include "utils/graph/multidigraph/multidigraph_view.h"

namespace FlexFlow {

struct MultiDiGraph : virtual public MultiDiGraphView {
  Node add_node();
  MultiDiEdge add_edge(Node const &, Node const &);

  void remove_node(Node const &);
  void remove_edge(MultiDiEdge const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &) const;
  Node get_multidiedge_src(MultiDiEdge const &) const;
  Node get_multidiedge_dst(MultiDiEdge const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IMultiDiGraph, T>::value,
                                 MultiDiGraph>::type
      create() {
    return MultiDiGraph(make_cow_ptr<T>());
  }

  template <typename T>
  static std::enable_if_t<std::is_base_of<IMultiDiGraph, T>::value,
                          MultiDiGraph>
      materialize_copy_of(MultiDiGraphView const &view) {
    cow_ptr_t<T> impl = make_cow_ptr<T>();
    impl.get_mutable()->inplace_materialize_from(view);
    return MultiDiGraph(std::move(impl));
  }

protected:
  using MultiDiGraphView::MultiDiGraphView;

private:
  IMultiDiGraph &get_interface();
  IMultiDiGraph const &get_interface() const;

  friend struct GraphInternal;
};

} // namespace FlexFlow

#endif

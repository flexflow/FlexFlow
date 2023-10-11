#ifndef _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H

#include "multidigraph.h"
#include "node.h"
#include "open_edge.h"
#include "open_graph_interfaces.h"
#include "utils/optional.h"
#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct OpenMultiDiGraphView : virtual MultiDiGraphView {
public:
  using Edge = OpenMultiDiEdge;
  using EdgeQuery = OpenMultiDiEdgeQuery;

  // OpenMultiDiGraphView() = delete;
  OpenMultiDiGraphView(OpenMultiDiGraphView const &) = default;
  OpenMultiDiGraphView &operator=(OpenMultiDiGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<IOpenMultiDiGraphView, T>::value,
                              OpenMultiDiGraphView>::type
      create(Args &&...args) {
    return OpenMultiDiGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using MultiDiGraphView::MultiDiGraphView;

private:
  cow_ptr_t<IOpenMultiDiGraphView> get_ptr() const;

  friend struct GraphInternal;
};

struct OpenMultiDiGraph : virtual OpenMultiDiGraphView {
public:
  using Edge = OpenMultiDiEdge;
  using EdgeQuery = OpenMultiDiEdgeQuery;

  OpenMultiDiGraph() = delete;
  OpenMultiDiGraph(OpenMultiDiGraph const &) = default;

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IOpenMultiDiGraph, T>::value,
                                 OpenMultiDiGraph>::type
      create() {
    return make_cow_ptr<T>();
  }

private:
  using OpenMultiDiGraphView::OpenMultiDiGraphView;

  cow_ptr_t<IOpenMultiDiGraph> get_ptr() const;

  friend struct GraphInternal;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(OpenMultiDiGraph);

struct UpwardOpenMultiDiGraphView : virtual MultiDiGraphView {
public:
  using Edge = UpwardOpenMultiDiEdge;
  using EdgeQuery = UpwardOpenMultiDiEdgeQuery;

  // UpwardOpenMultiDiGraphView() = delete;
  UpwardOpenMultiDiGraphView(UpwardOpenMultiDiGraphView const &) = default;
  UpwardOpenMultiDiGraphView &
      operator=(UpwardOpenMultiDiGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &);
  std::unordered_set<Edge> query_edges(EdgeQuery const &);

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<IUpwardOpenMultiDiGraphView, T>::value,
      UpwardOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return UpwardOpenMultiDiGraphView(
        cow_ptr_t<T>(std::forward<Args>(args)...));
  }

private:
  using MultiDiGraphView::MultiDiGraphView;

  cow_ptr_t<IUpwardOpenMultiDiGraphView> get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UpwardOpenMultiDiGraphView);

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

  cow_ptr_t<IUpwardOpenMultiDiGraph> get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UpwardOpenMultiDiGraph);

struct DownwardOpenMultiDiGraphView : virtual MultiDiGraphView {
public:
  using Edge = DownwardOpenMultiDiEdge;
  using EdgeQuery = DownwardOpenMultiDiEdgeQuery;
  using Interface = IDownwardOpenMultiDiGraphView;

  // DownwardOpenMultiDiGraphView() = delete;
  DownwardOpenMultiDiGraphView(DownwardOpenMultiDiGraphView const &) = default;
  DownwardOpenMultiDiGraphView &
      operator=(DownwardOpenMultiDiGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<IDownwardOpenMultiDiGraphView, T>::value,
      DownwardOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return DownwardOpenMultiDiGraphView(
        make_cow_ptr<T>(std::forward<Args>(args)...));
  }

private:
  using MultiDiGraphView::MultiDiGraphView;

  cow_ptr_t<Interface> get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DownwardOpenMultiDiGraphView);

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

  cow_ptr_t<IDownwardOpenMultiDiGraph> get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DownwardOpenMultiDiGraph);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H

#include "multidigraph.h"
#include "node.h"
#include "open_graph_interfaces.h"
#include "utils/optional.h"
#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct OpenMultiDiGraphView : MultiDiGraphView {
public:
  using Edge = OpenMultiDiEdge;
  using EdgeQuery = OpenMultiDiEdgeQuery;

  OpenMultiDiGraphView() = delete;

  friend void swap(OpenMultiDiGraphView &, OpenMultiDiGraphView &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<IOpenMultiDiGraphView, T>::value,
                              OpenMultiDiGraphView>::type
      create(Args &&...args) {
    return OpenMultiDiGraphView(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  OpenMultiDiGraphView(std::shared_ptr<IOpenMultiDiGraphView const> ptr);
  std::shared_ptr<IOpenMultiDiGraphView const> get_ptr() const;

  friend struct GraphInternal;
};

struct OpenMultiDiGraph {
public:
  using Edge = OpenMultiDiEdge;
  using EdgeQuery = OpenMultiDiEdgeQuery;

  OpenMultiDiGraph() = delete;
  OpenMultiDiGraph(OpenMultiDiGraph const &);

  friend void swap(OpenMultiDiGraph &, OpenMultiDiGraph &);

  operator OpenMultiDiGraphView() const;

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
  OpenMultiDiGraph(cow_ptr_t<IOpenMultiDiGraph> ptr);
  cow_ptr_t<IOpenMultiDiGraph> ptr;

  friend struct GraphInternal;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(OpenMultiDiGraph);

struct UpwardOpenMultiDiGraphView : virtual MultiDiGraphView {
public:
  using Edge = UpwardOpenMultiDiEdge;
  using EdgeQuery = UpwardOpenMultiDiEdgeQuery;

  UpwardOpenMultiDiGraphView() = delete;

  friend void swap(UpwardOpenMultiDiGraphView &, UpwardOpenMultiDiGraphView &);

  std::unordered_set<Node> query_nodes(NodeQuery const &);
  std::unordered_set<Edge> query_edges(EdgeQuery const &);

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<IUpwardOpenMultiDiGraphView, T>::value,
      UpwardOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return UpwardOpenMultiDiGraphView(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  UpwardOpenMultiDiGraphView(
      std::shared_ptr<IUpwardOpenMultiDiGraphView const>);

private:
  std::shared_ptr<IUpwardOpenMultiDiGraphView const> get_ptr();
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UpwardOpenMultiDiGraphView);

struct UpwardOpenMultiDiGraph {
public:
  using Edge = UpwardOpenMultiDiEdge;
  using EdgeQuery = UpwardOpenMultiDiEdgeQuery;

  UpwardOpenMultiDiGraph() = delete;
  UpwardOpenMultiDiGraph(UpwardOpenMultiDiGraph const &);

  UpwardOpenMultiDiGraph &operator=(UpwardOpenMultiDiGraph);

  friend void swap(UpwardOpenMultiDiGraph &, UpwardOpenMultiDiGraph &);

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
    return UpwardOpenMultiDiGraph(make_unique<T>());
  }

private:
  UpwardOpenMultiDiGraph(std::unique_ptr<IUpwardOpenMultiDiGraph>);

private:
  cow_ptr_t<IUpwardOpenMultiDiGraph> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UpwardOpenMultiDiGraph);

struct DownwardOpenMultiDiGraphView : virtual OpenMultiDiSubgraphView {
public:
  using Edge = DownwardOpenMultiDiEdge;
  using EdgeQuery = DownwardOpenMultiDiEdgeQuery;
  using Interface = IDownwardOpenMultiDiGraphView;

  DownwardOpenMultiDiGraphView() = delete;

  friend void swap(OpenMultiDiGraphView &, OpenMultiDiGraphView &);

  std::unordered_set<Node> query_nodes(NodeQuery const &);
  std::unordered_set<Edge> query_edges(EdgeQuery const &);

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<IDownwardOpenMultiDiGraphView, T>::value,
      DownwardOpenMultiDiGraphView>::type
      create(Args &&...args) {
    return DownwardOpenMultiDiGraphView(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  DownwardOpenMultiDiGraphView(
      std::shared_ptr<Interface const>);

private:
  std::shared_ptr<Interface const> get_ptr();
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DownwardOpenMultiDiGraphView);

struct DownwardOpenMultiDiGraph {
public:
  using Edge = DownwardOpenMultiDiEdge;
  using EdgeQuery = DownwardOpenMultiDiEdgeQuery;

  DownwardOpenMultiDiGraph() = delete;
  DownwardOpenMultiDiGraph(DownwardOpenMultiDiGraph const &);

  DownwardOpenMultiDiGraph &operator=(DownwardOpenMultiDiGraph);

  friend void swap(DownwardOpenMultiDiGraph &, DownwardOpenMultiDiGraph &);

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<
      std::is_base_of<IDownwardOpenMultiDiGraph, T>::value,
      DownwardOpenMultiDiGraph>::type
      create() {
    return DownwardOpenMultiDiGraph(make_unique<T>());
  }

private:
  DownwardOpenMultiDiGraph(std::unique_ptr<IDownwardOpenMultiDiGraph>);

private:
  cow_ptr_t<IDownwardOpenMultiDiGraph> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DownwardOpenMultiDiGraph);

} // namespace FlexFlow

#endif

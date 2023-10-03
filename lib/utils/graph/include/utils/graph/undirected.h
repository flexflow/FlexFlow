#ifndef _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H
#define _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H

#include "cow_ptr_t.h"
#include "node.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include "utils/type_traits.h"
#include "utils/unique.h"
#include <unordered_set>

namespace FlexFlow {

struct IUndirectedGraphView : public IGraphView {
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  IUndirectedGraphView(IUndirectedGraphView const &) = delete;
  IUndirectedGraphView &operator=(IUndirectedGraphView const &) = delete;

  virtual std::unordered_set<Edge>
      query_edges(UndirectedEdgeQuery const &) const = 0;
  virtual ~IUndirectedGraphView() = default;

protected:
  IUndirectedGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUndirectedGraphView);

struct UndirectedGraphView : virtual GraphView {
public:
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  UndirectedGraphView() = delete;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &query) const;

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<IUndirectedGraphView, T>::value,
                              UndirectedGraphView>::type
      create(Args &&...args) {
    return UndirectedGraphView(
        make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  UndirectedGraphView(cow_ptr_t<IUndirectedGraphView> ptr);

  friend struct GraphInternal;

private:
  cow_ptr_t<IUndirectedGraphView> get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UndirectedGraphView);

struct IUndirectedGraph : public IUndirectedGraphView {
  virtual void add_edge(UndirectedEdge const &) = 0;
  virtual void remove_edge(UndirectedEdge const &) = 0;

  virtual std::unordered_set<Node>
      query_nodes(NodeQuery const &query) const = 0;

  virtual IUndirectedGraph *clone() const override = 0;
};

struct UndirectedGraph : virtual UndirectedGraphView {
public:
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  UndirectedGraph() = delete;
  UndirectedGraph(UndirectedGraph const &) = default;
  UndirectedGraph &operator=(UndirectedGraph const &) = default;

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IUndirectedGraph, T>::value,
                                 UndirectedGraph>::type
      create() {
    return UndirectedGraph(make_cow_ptr<T>());
  }

protected:
  UndirectedGraph(cow_ptr_t<IUndirectedGraph>);

  friend struct GraphInternal;

private:
  cow_ptr_t<IUndirectedGraph> get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UndirectedGraph);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H
#define _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H

#include "cow_ptr_t.h"
#include "node.h"
#include "utils/optional.h"
#include "utils/unique.h"
#include <unordered_set>
#include "undirected_edge.h"
#include "undirected_interfaces.h"

namespace FlexFlow {

struct UndirectedEdgeQuery {
  query_set<Node> nodes;

  static UndirectedEdgeQuery all() {
    NOT_IMPLEMENTED();
  }
};
FF_VISITABLE_STRUCT(UndirectedEdgeQuery, nodes);

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &,
                                       UndirectedEdgeQuery const &);

struct IUndirectedGraphView : public IGraphView {
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  IUndirectedGraphView(IUndirectedGraphView const &) = delete;
  IUndirectedGraphView &operator=(IUndirectedGraphView const &) = delete;

  virtual std::unordered_set<Edge>
      query_edges(UndirectedEdgeQuery const &) const = 0;
  virtual ~IUndirectedGraphView();

protected:
  IUndirectedGraphView() = default;
};

static_assert(is_rc_copy_virtual_compliant<IUndirectedGraphView>::value,
              RC_COPY_VIRTUAL_MSG);

struct UndirectedGraphView {
public:
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  UndirectedGraphView() = delete;

  operator GraphView const &() const;
  operator GraphView &();

  friend void swap(UndirectedGraphView &, UndirectedGraphView &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  IUndirectedGraphView const *unsafe() const {
    return this->ptr.get();
  }

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<IUndirectedGraphView, T>::value,
                              UndirectedGraphView>::type
      create(Args &&...args) {
    return UndirectedGraphView(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  UndirectedGraphView(std::shared_ptr<IUndirectedGraphView const>);

  friend UndirectedGraphView unsafe(IUndirectedGraphView const &);

private:
  std::shared_ptr<IUndirectedGraphView const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UndirectedGraphView);

struct IUndirectedGraph : public IUndirectedGraphView, public IGraph {
  virtual void add_edge(UndirectedEdge const &) = 0;
  virtual void remove_edge(UndirectedEdge const &) = 0;

  virtual IUndirectedGraph *clone() const = 0;
};

struct UndirectedGraph {
public:
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  UndirectedGraph() = delete;
  UndirectedGraph(UndirectedGraph const &);

  UndirectedGraph &operator=(UndirectedGraph);

  operator UndirectedGraphView() const;

  friend void swap(UndirectedGraph &, UndirectedGraph &);

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IUndirectedGraph, T>::value,
                                 UndirectedGraph>::type
      create() {
    return UndirectedGraph(make_unique<T>());
  }

private:
  UndirectedGraph(std::unique_ptr<IUndirectedGraph>);

private:
  cow_ptr_t<IUndirectedGraph> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(UndirectedGraph);

} // namespace FlexFlow

#endif

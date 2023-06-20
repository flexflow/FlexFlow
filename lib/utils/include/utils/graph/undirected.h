#ifndef _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H
#define _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H

#include "cow_ptr_t.h"
#include "node.h"
#include "tl/optional.hpp"
#include "utils/maybe_owned_ref.h"
#include "utils/unique.h"
#include <unordered_set>

namespace FlexFlow {

struct UndirectedEdge : public use_visitable_cmp<UndirectedEdge> {
public:
  UndirectedEdge() = delete;
  UndirectedEdge(Node src, Node dst);

public:
  Node smaller, bigger;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::UndirectedEdge, smaller, bigger);
MAKE_VISIT_HASHABLE(::FlexFlow::UndirectedEdge);

namespace FlexFlow {

struct UndirectedEdgeQuery {
  UndirectedEdgeQuery() = delete;
  UndirectedEdgeQuery(optional<std::unordered_set<Node>> const &);

  optional<std::unordered_set<Node>> nodes = nullopt;
};

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &,
                                       UndirectedEdgeQuery const &);

struct IUndirectedGraphView : public IGraphView {
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  IUndirectedGraphView(IUndirectedGraphView const &) = delete;
  IUndirectedGraphView &operator=(IUndirectedGraphView const &) = delete;

  virtual std::unordered_set<Edge>
      query_edges(UndirectedEdgeQuery const &) const = 0;
  virtual ~IUndirectedGraphView()=default;

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

  operator GraphView () {
    return GraphView(this->ptr.get());
  }

  friend void swap(UndirectedGraphView &, UndirectedGraphView &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  operator maybe_owned_ref<IUndirectedGraphView const>() const {
    return maybe_owned_ref<IUndirectedGraphView const>(this->ptr.get());
  }

  IUndirectedGraphView const *unsafe() const {
    return this->ptr.get().get();
  }

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<IUndirectedGraphView, T>::value,
                              UndirectedGraphView>::type
      create(Args &&...args) {
    return UndirectedGraphView(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

  UndirectedGraphView(std::shared_ptr<IUndirectedGraphView const> ptr):ptr(ptr) {}

private:
  //friend UndirectedGraphView unsafe(IUndirectedGraphView const &);

private:
  cow_ptr_t<IUndirectedGraphView const> ptr;
  //std::shared_ptr<IUndirectedGraphView const> ptr;
};

//UndirectedGraphView unsafe(IUndirectedGraphView const &);

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

  operator UndirectedGraphView() const {
    return UndirectedGraphView(std::shared_ptr<IUndirectedGraphView const>(ptr->clone()));
  }


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
  std::unique_ptr<IUndirectedGraph> ptr;
};

static_assert(std::is_copy_constructible<UndirectedGraph>::value, "");
static_assert(std::is_move_constructible<UndirectedGraph>::value, "");
static_assert(std::is_copy_assignable<UndirectedGraph>::value, "");
static_assert(std::is_move_assignable<UndirectedGraph>::value, "");

} // namespace FlexFlow

#endif

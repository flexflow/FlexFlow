#ifndef _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H

#include "node.h"
#include "tl/optional.hpp"
#include <memory>
#include <unordered_set>
#include "utils/visitable.h"
#include "utils/unique.h"
#include "utils/maybe_owned_ref.h"

namespace FlexFlow {

struct DirectedEdge : use_visitable_cmp<DirectedEdge> {
public:
  DirectedEdge() = delete;
  DirectedEdge(Node src, Node dst);
public:
  Node src, dst;
};
std::ostream &operator<<(std::ostream &, DirectedEdge const &);

}

VISITABLE_STRUCT(::FlexFlow::DirectedEdge, src, dst);
MAKE_VISIT_HASHABLE(::FlexFlow::DirectedEdge);

namespace FlexFlow {

struct DirectedEdgeQuery {
  DirectedEdgeQuery() = default;
  DirectedEdgeQuery(tl::optional<std::unordered_set<Node>> const &srcs, tl::optional<std::unordered_set<Node>> const &dsts);
  tl::optional<std::unordered_set<Node>> srcs = tl::nullopt, 
                                         dsts = tl::nullopt;
};

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &, DirectedEdgeQuery const &);

struct IDiGraphView : public IGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  IDiGraphView(IDiGraphView const &) = delete;
  IDiGraphView &operator=(IDiGraphView const &) = delete;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual ~IDiGraphView()=default;
protected:
  IDiGraphView() = default;
};

static_assert(is_rc_copy_virtual_compliant<IDiGraphView>::value, RC_COPY_VIRTUAL_MSG);

struct DiGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraphView() = delete;

  operator GraphView() const;//TODO

  friend void swap(DiGraphView &, DiGraphView &);

  bool operator==(DiGraphView const &) const;
  bool operator!=(DiGraphView const &) const;

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  operator maybe_owned_ref<IDiGraphView const>() const {
    return maybe_owned_ref<IDiGraphView const>(this->ptr);
  }

  IDiGraphView const *unsafe() const {
    return this->ptr.get(); 
  }

  template <typename T, typename ...Args>
  static
  typename std::enable_if<std::is_base_of<IDiGraphView, T>::value, DiGraphView>::type
  create(Args &&... args) {
    return DiGraphView(std::make_shared<T>(std::forward<Args>(args)...));
  }
  DiGraphView(std::unique_ptr<IDiGraphView> const );

private:
  DiGraphView(std::shared_ptr<IDiGraphView const> ptr):ptr(ptr){}

  friend DiGraphView unsafe(IDiGraphView const &);
private:
  std::shared_ptr<IDiGraphView const> ptr;
};

DiGraphView unsafe(IDiGraphView const &);

struct IDiGraph : public IDiGraphView, public IGraph {
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;
  virtual IDiGraph *clone() const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IDiGraph>::value, RC_COPY_VIRTUAL_MSG);

struct DiGraph {
public: 
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraph() = delete;
  DiGraph(DiGraph const &);

  DiGraph &operator=(DiGraph);

  operator DiGraphView() const; //TODO

  friend void swap(DiGraph &, DiGraph &);

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static 
  typename std::enable_if<std::is_base_of<IDiGraph, T>::value, DiGraph>::type 
  create() { 
    return DiGraph(make_unique<T>());
  }
private:
  DiGraph(std::unique_ptr<IDiGraph>);
private:
  std::unique_ptr<IDiGraph> ptr;
};

static_assert(std::is_copy_constructible<DiGraph>::value, "");
static_assert(std::is_move_constructible<DiGraph>::value, "");
static_assert(std::is_copy_assignable<DiGraph>::value, "");
static_assert(std::is_move_assignable<DiGraph>::value, "");

}

#endif

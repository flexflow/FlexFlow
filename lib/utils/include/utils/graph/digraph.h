#ifndef _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H

#include "cow_ptr_t.h"
#include "node.h"
#include "tl/optional.hpp"
#include "utils/unique.h"
#include "utils/visitable.h"
#include <unordered_set>

namespace FlexFlow {

struct DirectedEdge {
  Node src;
  Node dst;
};

std::ostream &operator<<(std::ostream &s, DirectedEdge const &e);

FF_VISITABLE_STRUCT(DirectedEdge, src, dst);

struct DirectedEdgeQuery {
  query_set<Node> srcs;
  query_set<Node> dsts;

  static DirectedEdgeQuery all();
};
FF_VISITABLE_STRUCT(DirectedEdgeQuery, srcs, dsts);

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &,
                                     DirectedEdgeQuery const &);

struct IDiGraphView : public IGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  IDiGraphView(IDiGraphView const &) = delete;
  IDiGraphView &operator=(IDiGraphView const &) = delete;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual ~IDiGraphView() = default;

protected:
  IDiGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDiGraphView);

struct DiGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraphView() = delete;

  operator GraphView() const;

  friend void swap(DiGraphView &, DiGraphView &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;
  friend bool is_ptr_equal(DiGraphView const &, DiGraphView const &);
  IDiGraphView const *unsafe() const {
    return this->ptr.get();
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IDiGraphView, T>::value,
                                 DiGraphView>::type
      create(Args &&...args) {
    return DiGraphView(std::make_shared<T>(std::forward<Args>(args)...));
  }
  static DiGraphView unsafe_create(IDiGraphView const &graphView);


  DiGraphView(std::shared_ptr<IDiGraphView const> const &ptr,
              should_only_be_used_internally_tag_t const &tag)
      : DiGraphView(ptr) {}
private:
  DiGraphView(std::shared_ptr<IDiGraphView const> ptr) : ptr(ptr) {}
  std::shared_ptr<IDiGraphView const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DiGraphView);

struct IDiGraph : public IDiGraphView, public IGraph {
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;
  virtual IDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDiGraph);

struct DiGraph {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraph() = delete;
  DiGraph(DiGraph const &) = default;
  DiGraph &operator=(DiGraph const &) = default;

  operator DiGraphView() const {
    return DiGraphView(this->ptr.get(), should_only_be_used_internally_tag_t{});
  }

  friend void swap(DiGraph &, DiGraph &);

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IDiGraph, T>::value,
                                 DiGraph>::type
      create() {
    return DiGraph(make_unique<T>());
  }

private:
  DiGraph(std::unique_ptr<IDiGraph> ptr);
  cow_ptr_t<IDiGraph> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DiGraph);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H

#include "cow_ptr_t.h"
#include "node.h"
#include "tl/optional.hpp"
#include "utils/maybe_owned_ref.h"
#include "utils/type_traits.h"
#include "utils/unique.h"
#include "utils/visitable.h"
#include <memory>
#include <unordered_set>

namespace FlexFlow {

struct MultiDiEdge : use_visitable_cmp<MultiDiEdge> {
public:
  MultiDiEdge() = delete;
  MultiDiEdge(Node src, Node dst, size_t srcIdx, size_t dstIdx);

public:
  Node src, dst;
  std::size_t srcIdx, dstIdx;
};
std::ostream &operator<<(std::ostream &, MultiDiEdge const &);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::MultiDiEdge, src, dst, srcIdx, dstIdx);
MAKE_VISIT_HASHABLE(::FlexFlow::MultiDiEdge);

namespace FlexFlow {

struct MultiDiEdgeQuery {
  tl::optional<std::unordered_set<Node>> srcs = tl::nullopt, dsts = tl::nullopt;
  tl::optional<std::unordered_set<std::size_t>> srcIdxs = tl::nullopt,
                                                dstIdxs = tl::nullopt;

  MultiDiEdgeQuery(
      tl::optional<std::unordered_set<Node>> const &srcs = tl::nullopt,
      tl::optional<std::unordered_set<Node>> const &dsts = tl::nullopt,
      tl::optional<std::unordered_set<std::size_t>> const &srcIdxs =
          tl::nullopt,
      tl::optional<std::unordered_set<std::size_t>> const &dstIdxs =
          tl::nullopt);

  MultiDiEdgeQuery with_src_nodes(std::unordered_set<Node> const &) const;
  MultiDiEdgeQuery with_src_node(Node const &) const;
  MultiDiEdgeQuery with_dst_nodes(std::unordered_set<Node> const &) const;
  MultiDiEdgeQuery with_dst_node(Node const &) const;
  MultiDiEdgeQuery with_src_idxs(std::unordered_set<std::size_t> const &) const;
  MultiDiEdgeQuery with_src_idx(std::size_t) const;
  MultiDiEdgeQuery with_dst_idxs(std::unordered_set<std::size_t> const &) const;
  MultiDiEdgeQuery with_dst_idx(std::size_t) const;

  static MultiDiEdgeQuery all();
};

MultiDiEdgeQuery query_intersection(MultiDiEdgeQuery const &,
                                    MultiDiEdgeQuery const &);

struct IMultiDiGraphView : public IGraphView {
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual ~IMultiDiGraphView() = default;
};

static_assert(is_rc_copy_virtual_compliant<IMultiDiGraphView>::value,
              RC_COPY_VIRTUAL_MSG);

struct IMultiDiGraph : public IMultiDiGraphView, public IGraph {
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;

  virtual std::unordered_set<Node>
      query_nodes(NodeQuery const &query) const override {
    return static_cast<IMultiDiGraphView const *>(this)->query_nodes(query);
  }

  virtual IMultiDiGraph *clone() const override = 0;
};

static_assert(is_rc_copy_virtual_compliant<IMultiDiGraph>::value,
              RC_COPY_VIRTUAL_MSG);

struct MultiDiGraphView {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  operator GraphView() const {
    return GraphView(ptr);
  }

  friend void swap(MultiDiGraphView &, MultiDiGraphView &);

  operator maybe_owned_ref<IMultiDiGraphView const>() const {
    return maybe_owned_ref<IMultiDiGraphView const>(this->ptr);
  }

  IMultiDiGraphView const *unsafe() const {
    return this->ptr.get();
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IMultiDiGraphView, T>::value,
                                 MultiDiGraphView>::type
      create(Args &&...args) {
    return MultiDiGraphView(
        std::make_shared<T const>(std::forward<Args>(args)...));
  }

  MultiDiGraphView(std::shared_ptr<IMultiDiGraphView const> ptr) : ptr(ptr) {}

private:
  friend struct MultiDiGraph;
  friend MultiDiGraphView unsafe(IMultiDiGraphView const &);

private:
  std::shared_ptr<IMultiDiGraphView const> ptr;
};

MultiDiGraphView unsafe(IMultiDiGraphView const &graphView);

struct MultiDiGraph {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  MultiDiGraph() = delete;
  MultiDiGraph(MultiDiGraph const &);

  operator MultiDiGraphView() const;

  MultiDiGraph &operator=(MultiDiGraph);

  friend void swap(MultiDiGraph &, MultiDiGraph &);

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &e);
  void remove_edge(Edge const &e);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IMultiDiGraph, T>::value,
                                 MultiDiGraph>::type
      create() {
    return MultiDiGraph(make_unique<T>());
  }

private:
  MultiDiGraph(std::unique_ptr<IMultiDiGraph>);

private:
  cow_ptr_t<IMultiDiGraph> ptr;
};

static_assert(std::is_copy_constructible<MultiDiGraph>::value, "");
static_assert(std::is_move_constructible<MultiDiGraph>::value, "");
static_assert(std::is_copy_assignable<MultiDiGraph>::value, "");
static_assert(std::is_move_assignable<MultiDiGraph>::value, "");

} // namespace FlexFlow

#endif

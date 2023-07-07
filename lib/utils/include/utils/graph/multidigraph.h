#ifndef _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H

#include "cow_ptr_t.h"
#include "node.h"
#include "tl/optional.hpp"
#include "utils/type_traits.h"
#include "utils/unique.h"
#include "utils/visitable.h"
#include <unordered_set>

namespace FlexFlow {

/**
 * @class NodePort
 * @brief An opaque object used to disambiguate multiple edges between the same
 * nodes in a MultiDiGraph
 *
 * Name chosen to match the terminology used by <a href="linkURL">ELK</a>
 *
 */
struct NodePort : public strong_typedef<NodePort, size_t> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

MAKE_TYPEDEF_HASHABLE(::FlexFlow::NodePort);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::NodePort, "NodePort");

namespace FlexFlow {

struct MultiDiEdge {
  Node src, dst;
  NodePort srcIdx, dstIdx;
};



FF_VISITABLE_STRUCT(MultiDiEdge, src, dst, srcIdx, dstIdx);


struct MultiDiEdgeQuery {
  tl::optional<std::unordered_set<Node>> srcs = tl::nullopt, dsts = tl::nullopt;
  tl::optional<std::unordered_set<NodePort>> srcIdxs = tl::nullopt,
                                             dstIdxs = tl::nullopt;

  MultiDiEdgeQuery(
      tl::optional<std::unordered_set<Node>> const &srcs = tl::nullopt,
      tl::optional<std::unordered_set<Node>> const &dsts = tl::nullopt,
      tl::optional<std::unordered_set<NodePort>> const &srcIdxs = tl::nullopt,
      tl::optional<std::unordered_set<NodePort>> const &dstIdxs = tl::nullopt);

  MultiDiEdgeQuery with_src_nodes(std::unordered_set<Node> const &) const;
  MultiDiEdgeQuery with_src_node(Node const &) const;
  MultiDiEdgeQuery with_dst_nodes(std::unordered_set<Node> const &) const;
  MultiDiEdgeQuery with_dst_node(Node const &) const;
  MultiDiEdgeQuery with_src_idxs(std::unordered_set<NodePort> const &) const;
  MultiDiEdgeQuery with_src_idx(NodePort const &) const;
  MultiDiEdgeQuery with_dst_idxs(std::unordered_set<NodePort> const &) const;
  MultiDiEdgeQuery with_dst_idx(NodePort const &) const;

  static MultiDiEdgeQuery all();
};

struct MultiDiInput {
  Node node;
  NodePort idx;
};
FF_VISITABLE_STRUCT(MultiDiInput, node, idx);

MultiDiInput get_input(MultiDiEdge const &);

struct MultiDiOutput {
  Node node;
  NodePort idx;
};
FF_VISITABLE_STRUCT(MultiDiOutput, node, idx);

MultiDiOutput get_output(MultiDiEdge const &);

MultiDiEdgeQuery query_intersection(MultiDiEdgeQuery const &,
                                    MultiDiEdgeQuery const &);

struct IMultiDiGraphView : public IGraphView {
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual ~IMultiDiGraphView()=default;
};

static_assert(is_rc_copy_virtual_compliant<IMultiDiGraphView>::value,
              RC_COPY_VIRTUAL_MSG);

struct IMultiDiGraph : public IMultiDiGraphView, public IGraph {
  virtual NodePort add_node_port();
  virtual void add_node_port_unsafe(NodePort const &);
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;

  virtual std::unordered_set<Node>
      query_nodes(NodeQuery const &query) const override {
    return static_cast<IMultiDiGraphView const *>(this)->query_nodes(query);
  }

  virtual IMultiDiGraph *clone() const override = 0;  
  private:
    std::size_t next_node_idx = 0;
};

static_assert(is_rc_copy_virtual_compliant<IMultiDiGraph>::value,
              RC_COPY_VIRTUAL_MSG);

struct MultiDiGraphView {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  operator GraphView() const;

  friend void swap(MultiDiGraphView &, MultiDiGraphView &);

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
  MultiDiGraphView(std::shared_ptr<IMultiDiGraphView const> ptr):ptr(ptr){}
private:
  
  friend struct MultiDiGraph;
  friend MultiDiGraphView unsafe_create(IMultiDiGraphView const &);

private:
  std::shared_ptr<IMultiDiGraphView const> ptr;
};

MultiDiGraphView unsafe_create(IMultiDiGraphView const &);

struct MultiDiGraph {
public:
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  MultiDiGraph() = delete;
  MultiDiGraph(MultiDiGraph const &) = default;
  MultiDiGraph &operator=(MultiDiGraph const &) = default;

  operator MultiDiGraphView() const;


  friend void swap(MultiDiGraph &, MultiDiGraph &);

  Node add_node();
  NodePort add_node_port();
  void add_node_unsafe(Node const &);
  void add_node_port_unsafe(NodePort const &);
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

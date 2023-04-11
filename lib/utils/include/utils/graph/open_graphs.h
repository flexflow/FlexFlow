#ifndef _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H

#include "node.h"
#include "multidigraph.h"
#include "utils/variant.h"
#include "tl/optional.hpp"
#include "utils/visitable.h"

namespace FlexFlow {

struct InputMultiDiEdge {
  std::pair<std::size_t, std::size_t> uid; // necessary to differentiate multiple input edges from different sources resulting from a graph cut

  Node dst;
  std::size_t dstIdx;
};
bool operator==(InputMultiDiEdge const &, InputMultiDiEdge const &);

struct OutputMultiDiEdge {
  std::pair<std::size_t, std::size_t> uid; // necessary to differentiate multiple output edges from different sources resulting from a graph cut

  Node src;
  std::size_t srcIdx;
};
bool operator==(OutputMultiDiEdge const &, OutputMultiDiEdge const &);

using OpenMultiDiEdge = variant<
  InputMultiDiEdge,
  OutputMultiDiEdge,
  MultiDiEdge
>;

using DownwardOpenMultiDiEdge = variant<
  OutputMultiDiEdge,
  MultiDiEdge
>;

using UpwardOpenMultiDiEdge = variant<
  InputMultiDiEdge,
  MultiDiEdge
>;

bool is_input_edge(OpenMultiDiEdge const &);
bool is_output_edge(OpenMultiDiEdge const &);
bool is_standard_edge(OpenMultiDiEdge const &);

struct OutputMultiDiEdgeQuery {
  tl::optional<std::unordered_set<Node>> srcs = tl::nullopt;
  tl::optional<std::unordered_set<std::size_t>> srcIdxs = tl::nullopt;

  static OutputMultiDiEdgeQuery all();
  static OutputMultiDiEdgeQuery none();
};

struct InputMultiDiEdgeQuery {
  tl::optional<std::unordered_set<Node>> dsts = tl::nullopt;
  tl::optional<std::unordered_set<std::size_t>> dstIdxs = tl::nullopt;

  static InputMultiDiEdgeQuery all();
  static InputMultiDiEdgeQuery none();
};

struct OpenMultiDiEdgeQuery {
  InputMultiDiEdgeQuery input_edge_query;
  MultiDiEdgeQuery standard_edge_query;
  OutputMultiDiEdgeQuery output_edge_query;
};

struct DownwardOpenMultiDiEdgeQuery {
  OutputMultiDiEdgeQuery output_edge_query;
  MultiDiEdgeQuery standard_edge_query;
};

struct UpwardOpenMultiDiEdgeQuery {
  InputMultiDiEdgeQuery input_edge_query;
  MultiDiEdgeQuery standard_edge_query;
};

}

namespace std {

template <>
struct hash<::FlexFlow::OpenMultiDiEdge> {
  size_t operator()(::FlexFlow::OpenMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::DownwardOpenMultiDiEdge> {
  size_t operator()(::FlexFlow::DownwardOpenMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::UpwardOpenMultiDiEdge> {
  size_t operator()(::FlexFlow::UpwardOpenMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::OutputMultiDiEdge> {
  size_t operator()(::FlexFlow::OutputMultiDiEdge const &) const;
};

template <>
struct hash<::FlexFlow::InputMultiDiEdge> {
  size_t operator()(::FlexFlow::InputMultiDiEdge const &) const;
};

}

VISITABLE_STRUCT(::FlexFlow::InputMultiDiEdge, uid, dst, dstIdx);
VISITABLE_STRUCT(::FlexFlow::OutputMultiDiEdge, uid, src, srcIdx);

namespace FlexFlow {

struct IOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<OpenMultiDiEdge> query_edges(OpenMultiDiEdgeQuery const &) const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IOpenMultiDiGraphView>::value, RC_COPY_VIRTUAL_MSG);

struct IDownwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<DownwardOpenMultiDiEdge> query_edges(DownwardOpenMultiDiEdgeQuery const &) const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IDownwardOpenMultiDiGraphView>::value, RC_COPY_VIRTUAL_MSG);

struct IUpwardOpenMultiDiGraphView : public IGraphView {
  virtual std::unordered_set<UpwardOpenMultiDiEdge> query_edges(UpwardOpenMultiDiEdgeQuery const &) const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IUpwardOpenMultiDiGraphView>::value, RC_COPY_VIRTUAL_MSG);

struct IOpenMultiDiGraph : public IOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(OpenMultiDiEdge const &) = 0;
  virtual void remove_edge(OpenMultiDiEdge const &) = 0;
  virtual IOpenMultiDiGraph *clone() const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IOpenMultiDiGraph>::value, RC_COPY_VIRTUAL_MSG);

struct IUpwardOpenMultiDiGraph : public IUpwardOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(UpwardOpenMultiDiEdge const &) = 0;
  virtual IUpwardOpenMultiDiGraph *clone() const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IUpwardOpenMultiDiGraph>::value, RC_COPY_VIRTUAL_MSG);

struct IDownwardOpenMultiDiGraph : public IDownwardOpenMultiDiGraphView, public IGraph {
  virtual void add_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual void remove_edge(DownwardOpenMultiDiEdge const &) = 0;
  virtual IDownwardOpenMultiDiGraph *clone() const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IDownwardOpenMultiDiGraph>::value, RC_COPY_VIRTUAL_MSG);

struct OpenMultiDiGraph {
public:
  using Edge = OpenMultiDiEdge;
  using EdgeQuery = OpenMultiDiEdgeQuery;

  OpenMultiDiGraph() = delete;
  OpenMultiDiGraph(OpenMultiDiGraph const &);

  OpenMultiDiGraph &operator=(OpenMultiDiGraph);

  friend void swap(OpenMultiDiGraph &, OpenMultiDiGraph &);

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static 
  typename std::enable_if<std::is_base_of<IOpenMultiDiGraph, T>::value, OpenMultiDiGraph>::type 
  create() { 
    return OpenMultiDiGraph(make_unique<T>());
  }

private:
  OpenMultiDiGraph(std::unique_ptr<IOpenMultiDiGraph>);
private:
  std::unique_ptr<IOpenMultiDiGraph> ptr;
};

static_assert(std::is_copy_constructible<OpenMultiDiGraph>::value, "");
static_assert(std::is_move_constructible<OpenMultiDiGraph>::value, "");
static_assert(std::is_copy_assignable<OpenMultiDiGraph>::value, "");
static_assert(std::is_copy_constructible<OpenMultiDiGraph>::value, "");

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
  static 
  typename std::enable_if<std::is_base_of<IUpwardOpenMultiDiGraph, T>::value, UpwardOpenMultiDiGraph>::type 
  create() { 
    return UpwardOpenMultiDiGraph(make_unique<T>());
  }

private:
  UpwardOpenMultiDiGraph(std::unique_ptr<IUpwardOpenMultiDiGraph>);
private:
  std::unique_ptr<IUpwardOpenMultiDiGraph> ptr; 
};

static_assert(std::is_copy_constructible<UpwardOpenMultiDiGraph>::value, "");
static_assert(std::is_move_constructible<UpwardOpenMultiDiGraph>::value, "");
static_assert(std::is_copy_assignable<UpwardOpenMultiDiGraph>::value, "");
static_assert(std::is_copy_constructible<UpwardOpenMultiDiGraph>::value, "");

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
  static 
  typename std::enable_if<std::is_base_of<IDownwardOpenMultiDiGraph, T>::value, DownwardOpenMultiDiGraph>::type 
  create() { 
    return DownwardOpenMultiDiGraph(make_unique<T>());
  }

private:
  DownwardOpenMultiDiGraph(std::unique_ptr<IDownwardOpenMultiDiGraph>);
private:
  std::unique_ptr<IDownwardOpenMultiDiGraph> ptr; 
};

static_assert(std::is_copy_constructible<DownwardOpenMultiDiGraph>::value, "");
static_assert(std::is_move_constructible<DownwardOpenMultiDiGraph>::value, "");
static_assert(std::is_copy_assignable<DownwardOpenMultiDiGraph>::value, "");
static_assert(std::is_copy_constructible<DownwardOpenMultiDiGraph>::value, "");

}

#endif 

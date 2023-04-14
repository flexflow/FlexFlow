#ifndef _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_OPEN_GRAPHS_H

#include "node.h"
#include "multidigraph.h"
#include "utils/variant.h"
#include "tl/optional.hpp"
#include "utils/visitable.h"
#include "open_graph_interfaces.h"

namespace FlexFlow {

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

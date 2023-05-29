#ifndef _FLEXFLOW_UTILS_GRAPH_LABELLED_GRAPHS_H
#define _FLEXFLOW_UTILS_GRAPH_LABELLED_GRAPHS_H

#include <memory>
#include "multidigraph.h"
#include <unordered_map>
#include "open_graphs.h"
#include "utils/unique.h"
#include "utils/exception.h"
#include "labelled_graph_interfaces.h"

namespace FlexFlow {

template <typename NodeLabel>
struct NodeLabelledMultiDiGraph {
private:
  using Interface = INodeLabelledMultiDiGraph<NodeLabel>;
public:
  NodeLabelledMultiDiGraph() = delete;
  NodeLabelledMultiDiGraph(NodeLabelledMultiDiGraph const &other)
    : ptr(other.ptr->clone())
  { }
  NodeLabelledMultiDiGraph &operator=(NodeLabelledMultiDiGraph other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(NodeLabelledMultiDiGraph &lhs, NodeLabelledMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  Node add_node(NodeLabel const &l) { return this->ptr->add_node(l); }
  NodeLabel &at(Node const &n) { return this->ptr->at(n); }
  NodeLabel const &at(Node const &n) const { return this->ptr->at(n); }

  void add_edge(MultiDiEdge const &e) { return this->ptr->add_edge(e); }
  
  std::unordered_set<Node> query_nodes(NodeQuery const &q) const { return this->ptr->query_nodes(q); }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const { return this->ptr->query_edges(q); }

  template <typename BaseImpl>
  static 
  typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value, NodeLabelledMultiDiGraph>::type
  create() {
    return NodeLabelledMultiDiGraph(make_unique<BaseImpl>());
  }
private:
  NodeLabelledMultiDiGraph(std::unique_ptr<Interface> ptr) 
    : ptr(std::move(ptr)) 
  { }
private:
  std::unique_ptr<Interface> ptr; 
};

template <typename NodeLabel, typename EdgeLabel>
struct LabelledMultiDiGraph {
private:
  using Interface = ILabelledMultiDiGraph<NodeLabel, EdgeLabel>;
public:
  LabelledMultiDiGraph() = delete;
  LabelledMultiDiGraph(LabelledMultiDiGraph const &other)
    : ptr(other.ptr->clone())
  { }
  LabelledMultiDiGraph &operator=(LabelledMultiDiGraph other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(LabelledMultiDiGraph &lhs, LabelledMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  operator MultiDiGraphView() const;

  Node add_node(NodeLabel const &l) { return this->ptr->add_node(l); }
  NodeLabel &at(Node const &n) { return this->ptr->at(n); }
  // NodeLabel const &at(Node const &n) const { return this->ptr->at(n); }
  NodeLabel const &at(Node const &n) const;

  void add_edge(MultiDiEdge const &e, EdgeLabel const &l) { return this->ptr->add_edge(e, l); }
  EdgeLabel &at(MultiDiEdge const &e) { return this->ptr->at(e); }
  EdgeLabel const &at(MultiDiEdge const &e) const { return this->ptr->at(e); }
  
  std::unordered_set<Node> query_nodes(NodeQuery const &q) const { return this->ptr->query_nodes(q); }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const { return this->ptr->query_edges(q); }

  template <typename BaseImpl>
  static 
  typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value, LabelledMultiDiGraph>::type
  create() {
    return LabelledMultiDiGraph(make_unique<BaseImpl>());
  }
private:
  LabelledMultiDiGraph(std::unique_ptr<Interface> ptr)
    : ptr(std::move(ptr))
  { }
private:
  std::unique_ptr<Interface> ptr;
};

template <typename NodeLabel, typename OutputLabel>
struct OutputLabelledMultiDiGraph {
private:
  using Interface = IOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>;
public:
  OutputLabelledMultiDiGraph() = delete;
  OutputLabelledMultiDiGraph(OutputLabelledMultiDiGraph const &other)
    : ptr(other.ptr->clone()) { }
  OutputLabelledMultiDiGraph &operator=(OutputLabelledMultiDiGraph other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(OutputLabelledMultiDiGraph &lhs, OutputLabelledMultiDiGraph &rhs) { 
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  Node add_node(NodeLabel const &l) { return this->ptr->add_node(l); }
  NodeLabel &at(Node const &n) { return this->ptr->at(n); }
  NodeLabel const &at(Node const &n) const { return this->ptr->at(n); }

  void add_output(MultiDiOutput const &o, OutputLabel const &l) { return this->ptr->add_output(o, l); };
  void add_edge(MultiDiOutput const &o, MultiDiInput const &i) { return this->ptr->add_edge(o, i); };

  OutputLabel &at(MultiDiOutput const &o) { return this->ptr->at(o); }
  OutputLabel const &at(MultiDiOutput const &o) const { return this->ptr->at(o); }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const { return this->ptr->query_nodes(q); }
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const { return this->ptr->query_edges(q); }
private:
  OutputLabelledMultiDiGraph(std::unique_ptr<IOutputLabelledMultiDiGraph<NodeLabel, OutputLabel>> ptr)
    : ptr(std::move(ptr))
  { }
private:
  std::unique_ptr<Interface> ptr;
};

template<typename NodeLabel,
         typename EdgeLabel,
         typename InputLabel = EdgeLabel,
         typename OutputLabel = InputLabel>
struct LabelledOpenMultiDiGraphView {
public:
  LabelledOpenMultiDiGraphView() = delete;

  ILabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel, OutputLabel> const *unsafe() const {
    return this->ptr.get();
  }

private:
  std::shared_ptr<ILabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel, OutputLabel> const> ptr;
};

template<typename NodeLabel, 
         typename EdgeLabel, 
         typename InputLabel = EdgeLabel, 
         typename OutputLabel = InputLabel>
struct LabelledOpenMultiDiGraph {
private:
  using Interface = ILabelledOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel, OutputLabel>;
public:
  LabelledOpenMultiDiGraph() = delete;
  LabelledOpenMultiDiGraph(LabelledOpenMultiDiGraph const &other)
    : ptr(other.ptr->clone()) { }

  LabelledOpenMultiDiGraph& operator=(LabelledOpenMultiDiGraph const &other) {
    swap(*this, other);
    return *this;
  }

  operator OpenMultiDiGraph() const;
  operator LabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel, OutputLabel>() const;

  friend void swap(LabelledOpenMultiDiGraph &lhs, LabelledOpenMultiDiGraph &rhs) {
    using std::swap;

    swap(lhs.ptr, rhs.ptr);
  }

  Node add_node(NodeLabel const &l) { return this->ptr->add_node(l); }
  NodeLabel &at(Node const &n) { return this->ptr->at(n); }
  // NodeLabel const &at(Node const &n) const { return this->ptr->at(n); }
  NodeLabel const &at(Node const &n) const;

  void add_node_unsafe(Node const &n, NodeLabel const &l);

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const { return this->ptr->query_nodes(q); }
  std::unordered_set<OpenMultiDiEdge> query_edges(OpenMultiDiEdgeQuery const &q) const { return this->ptr->query_edges(q); }

  void add_edge(MultiDiEdge const &e, EdgeLabel const &l) { return this->ptr->add_edge(e, l); }
  EdgeLabel &at(MultiDiEdge const &e) { return this->ptr->at(e); }
  EdgeLabel const &at(MultiDiEdge const &e) const { return this->ptr->at(e); }

  void add_edge(InputMultiDiEdge const &e, InputLabel const &l) { return this->ptr->add_edge(e, l); }
  InputLabel &at(InputMultiDiEdge const &e) { return this->ptr->at(e); }
  InputLabel const &at(InputMultiDiEdge const &e) const { return this->ptr->at(e); }
  
  void add_edge(OutputMultiDiEdge const &, OutputLabel const &);
  OutputLabel &at(OutputMultiDiEdge const &);
  OutputLabel const &at(OutputMultiDiEdge const &) const;

  template <typename BaseImpl>
  static 
  typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value, LabelledOpenMultiDiGraph>::type
  create() {
    return LabelledOpenMultiDiGraph(make_unique<BaseImpl>());
  }
private:
  LabelledOpenMultiDiGraph(std::unique_ptr<Interface> ptr)
    : ptr(std::move(ptr))
  { }
private:
  std::unique_ptr<Interface> ptr;
};
static_assert(std::is_copy_constructible<NodeLabelledMultiDiGraph<int>>::value, "");
static_assert(std::is_move_constructible<NodeLabelledMultiDiGraph<int>>::value, "");
static_assert(std::is_copy_assignable<NodeLabelledMultiDiGraph<int>>::value, "");
static_assert(std::is_move_assignable<NodeLabelledMultiDiGraph<int>>::value, "");

static_assert(std::is_copy_constructible<LabelledMultiDiGraph<int, int>>::value, "");
static_assert(std::is_move_constructible<LabelledMultiDiGraph<int, int>>::value, "");
static_assert(std::is_copy_assignable<LabelledMultiDiGraph<int, int>>::value, "");
static_assert(std::is_move_assignable<LabelledMultiDiGraph<int, int>>::value, "");

static_assert(std::is_copy_constructible<OutputLabelledMultiDiGraph<int, int>>::value, "");
static_assert(std::is_move_constructible<OutputLabelledMultiDiGraph<int, int>>::value, "");
static_assert(std::is_copy_assignable<OutputLabelledMultiDiGraph<int, int>>::value, "");
static_assert(std::is_move_assignable<OutputLabelledMultiDiGraph<int, int>>::value, "");

static_assert(std::is_copy_constructible<LabelledOpenMultiDiGraph<int, int>>::value, "");
static_assert(std::is_move_constructible<LabelledOpenMultiDiGraph<int, int>>::value, "");
static_assert(std::is_copy_assignable<LabelledOpenMultiDiGraph<int, int>>::value, "");
static_assert(std::is_move_assignable<LabelledOpenMultiDiGraph<int, int>>::value, "");

}

#endif 

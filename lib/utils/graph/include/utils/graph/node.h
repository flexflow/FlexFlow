#ifndef _FLEXFLOW_UTILS_GRAPH_NODE_H
#define _FLEXFLOW_UTILS_GRAPH_NODE_H

#include "cow_ptr_t.h"
#include "query_set.h"
#include "utils/fmt.h"
#include "utils/optional.h"
#include "utils/strong_typedef.h"
#include "utils/type_traits.h"
#include "utils/unique.h"
#include "utils/visitable.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>
#include <unordered_set>

namespace FlexFlow {

struct Node : public strong_typedef<Node, size_t> {
  using strong_typedef::strong_typedef;
};
FF_TYPEDEF_HASHABLE(Node);
FF_TYPEDEF_PRINTABLE(Node, "Node");

struct NodeQuery {
  NodeQuery(query_set<Node> const &nodes) : nodes(nodes) {}

  query_set<Node> nodes;

  static NodeQuery all();
};
FF_VISITABLE_STRUCT(NodeQuery, nodes);

NodeQuery query_intersection(NodeQuery const &, NodeQuery const &);
NodeQuery query_union(NodeQuery const &, NodeQuery const &);

struct IGraphView {
  IGraphView() = default;
  IGraphView(IGraphView const &) = delete;
  IGraphView &operator=(IGraphView const &) = delete;

  virtual std::unordered_set<Node> query_nodes(NodeQuery const &) const = 0;
  virtual ~IGraphView(){};
};

struct GraphView {
  GraphView() = delete;

  friend void swap(GraphView &, GraphView &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IGraphView, T>::value,
                                 GraphView>::type
      create(Args &&...args) {
    return GraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

private:
  GraphView(cow_ptr_t<IGraphView> ptr);

  friend struct GraphInternal;

private:
  cow_ptr_t<IGraphView> ptr;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IGraphView);

struct IGraph : virtual IGraphView {
  IGraph() = default;
  IGraph(IGraph const &) = delete;
  IGraph &operator=(IGraph const &) = delete;

  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &) = 0;
  virtual void remove_node_unsafe(Node const &) = 0;
  virtual IGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IGraph);

struct Graph : virtual GraphView {
public:
  Graph() = delete;
  Graph(Graph const &) = default;

  Graph &operator=(Graph) = default;

  friend void swap(Graph &, Graph &);

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IGraph, T>::value, Graph>::type
      create() {
    return Graph(make_cow_ptr<T>());
  }

private:
  Graph(cow_ptr_t<IGraph>);
  cow_ptr_t<IGraph> get_ptr() const;

  friend struct GraphInternal;
};

} // namespace FlexFlow

#endif

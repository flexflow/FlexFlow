#ifndef _FLEXFLOW_UTILS_GRAPH_NODE_H
#define _FLEXFLOW_UTILS_GRAPH_NODE_H

#include <cstddef>
#include <functional>
#include <unordered_set>
#include "tl/optional.hpp"
#include <ostream>
#include "utils/visitable.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct Node {
public:
  Node() = delete;
  explicit Node(std::size_t idx); 

  bool operator==(Node const &) const;
  bool operator!=(Node const &) const;
  bool operator<(Node const &) const;

  std::string to_string() const;
public:
  std::size_t idx;
};
std::ostream &operator<<(std::ostream &, Node const &);

}

VISITABLE_STRUCT(::FlexFlow::Node, idx);

namespace std {
template <>
struct hash<::FlexFlow::Node> {
  std::size_t operator()(::FlexFlow::Node const &) const;
};
}

namespace FlexFlow {

struct NodeQuery {
  NodeQuery() = default;
  NodeQuery(std::unordered_set<Node> const &nodes);
  NodeQuery(tl::optional<std::unordered_set<Node>> const &nodes);

  tl::optional<std::unordered_set<Node>> nodes = tl::nullopt;
};

NodeQuery query_intersection(NodeQuery const &, NodeQuery const &);
NodeQuery query_union(NodeQuery const &, NodeQuery const &);

struct IGraphView {
  virtual std::unordered_set<Node> query_nodes(NodeQuery const &) const = 0;
};

struct IGraph {
  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &) = 0;
  virtual void remove_node_unsafe(Node const &) = 0;
};

}


#endif 

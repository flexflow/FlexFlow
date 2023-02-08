#ifndef _FLEXFLOW_UTILS_GRAPH_NODE_H
#define _FLEXFLOW_UTILS_GRAPH_NODE_H

#include <cstddef>
#include <functional>
#include <unordered_set>
#include "tl/optional.hpp"
#include <ostream>

namespace FlexFlow {
namespace utils {
namespace graph {

struct Node {
public:
  Node() = delete;
  explicit Node(std::size_t idx); 

  bool operator==(Node const &) const;
  bool operator!=(Node const &) const;
  bool operator<(Node const &) const;

  using AsConstTuple = std::tuple<size_t>;
  AsConstTuple as_tuple() const;

  std::string to_string() const;
public:
  std::size_t idx;
};
std::ostream &operator<<(std::ostream &, Node const &);

}
}
}

namespace std {
template <>
struct hash<::FlexFlow::utils::graph::Node> {
  std::size_t operator()(::FlexFlow::utils::graph::Node const &) const;
};
}

namespace FlexFlow {
namespace utils {
namespace graph {

struct NodeQuery {
  NodeQuery() = default;
  NodeQuery(std::unordered_set<Node> const &nodes);
  NodeQuery(tl::optional<std::unordered_set<Node>> const &nodes);

  tl::optional<std::unordered_set<Node>> nodes = tl::nullopt;
};

struct IGraphView {
  virtual std::unordered_set<Node> query_nodes(NodeQuery const &) const = 0;
};

struct IGraph {
  virtual Node add_node() = 0;
};
}
}
}



#endif 

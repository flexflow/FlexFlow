#ifndef _FLEXFLOW_UTILS_GRAPH_NODE_H
#define _FLEXFLOW_UTILS_GRAPH_NODE_H

#include <cstddef>
#include <functional>
#include <unordered_set>
#include <ostream>
#include "utils/visitable.h"
#include "utils/optional.h"
#include "utils/fmt.h"
#include <memory>
#include "utils/type_traits.h"

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
  IGraphView() = default;
  IGraphView(IGraphView const &) = delete;
  IGraphView &operator=(IGraphView const &) = delete;

  virtual std::unordered_set<Node> query_nodes(NodeQuery const &) const = 0;
  virtual ~IGraphView() {};
};

static_assert(is_rc_copy_virtual_compliant<IGraphView>::value, RC_COPY_VIRTUAL_MSG);

struct IGraph : IGraphView {
  IGraph(IGraph const &) = delete;
  IGraph &operator=(IGraph const &) = delete;

  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &) = 0;
  virtual void remove_node_unsafe(Node const &) = 0;
  virtual IGraph *clone() const = 0;
};

static_assert(is_rc_copy_virtual_compliant<IGraph>::value, RC_COPY_VIRTUAL_MSG);

}

namespace fmt {

template <> 
struct formatter<::FlexFlow::Node> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::Node const &n, FormatContext &ctx) const -> decltype(ctx.out()) {
    return formatter<std::string>(fmt::format("Node({})", n.idx), ctx);
  }
};

}


#endif 

#ifndef _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H

#include "tl/optional.hpp"
#include <unordered_set>
#include "node.h"

namespace FlexFlow {
namespace utils {
namespace graph {
namespace multidigraph {

struct Edge {
public:
  Edge() = delete;
  Edge(Node src, Node dst, size_t srcIdx, size_t dstIdx);

  bool operator==(Edge const &) const;
  bool operator<(Edge const &) const;

  using AsConstTuple = std::tuple<Node, Node, std::size_t, std::size_t>;
  AsConstTuple as_tuple() const;
public:
  Node src, dst;
  std::size_t srcIdx, dstIdx;
};
}
}
}
}

namespace std {
template <>
struct hash<::FlexFlow::utils::graph::multidigraph::Edge> {
  std::size_t operator()(::FlexFlow::utils::graph::multidigraph::Edge const &) const;
};
}


namespace FlexFlow {
namespace utils {
namespace graph {
namespace multidigraph {

struct EdgeQuery {
  tl::optional<std::unordered_set<Node>> srcs, dsts;
  tl::optional<std::unordered_set<std::size_t>> srcIdxs, dstIdxs;
};

struct IMultiDiGraph {
  virtual Node add_node() = 0;
  virtual void add_edge(Edge const &) = 0;
  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual std::unordered_set<Node> query_nodes(NodeQuery const &) const = 0;
};

}

using IMultiDiGraph = multidigraph::IMultiDiGraph;

}
}
}

#endif 

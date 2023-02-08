#ifndef _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H

#include "tl/optional.hpp"
#include <unordered_set>
#include "node.h"

namespace FlexFlow {
namespace utils {
namespace graph {
namespace multidigraph {

using Node = graph::Node;
using NodeQuery = graph::NodeQuery;

struct Edge {
public:
  Edge() = delete;
  Edge(Node src, Node dst, size_t srcIdx, size_t dstIdx);

  bool operator==(Edge const &) const;
  bool operator<(Edge const &) const;

  using AsConstTuple = std::tuple<Node, Node, std::size_t, std::size_t>;
  AsConstTuple as_tuple() const;

  std::string to_string() const;
public:
  Node src, dst;
  std::size_t srcIdx, dstIdx;
};
std::ostream &operator<<(std::ostream &, Edge const &);

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

  EdgeQuery with_src_nodes(std::unordered_set<Node> const &) const;
  EdgeQuery with_src_node(Node const &) const;
  EdgeQuery with_dst_nodes(std::unordered_set<Node> const &) const;
  EdgeQuery with_dst_node(Node const &) const;
  EdgeQuery with_src_idxs(std::unordered_set<std::size_t> const &) const;
  EdgeQuery with_src_idx(std::size_t) const;
  EdgeQuery with_dst_idxs(std::unordered_set<std::size_t> const &) const;
  EdgeQuery with_dst_idx(std::size_t) const;

  static EdgeQuery all();
};

struct IMultiDiGraphView : public IGraphView {
  using Edge = multidigraph::Edge;
  using EdgeQuery = multidigraph::EdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
};

struct IMultiDiGraph : public IMultiDiGraphView, public IGraph {
  virtual void add_edge(Edge const &) = 0;
};

}

using IMultiDiGraph = multidigraph::IMultiDiGraph;
using IMultiDiGraphView = multidigraph::IMultiDiGraphView;

}
}
}

#endif 

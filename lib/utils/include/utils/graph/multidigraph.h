#ifndef _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_MULTIDIGRAPH_H

#include "tl/optional.hpp"
#include <unordered_set>
#include "node.h"

namespace FlexFlow {
namespace utils {

struct MultiDiEdge {
public:
  MultiDiEdge() = delete;
  MultiDiEdge(Node src, Node dst, size_t srcIdx, size_t dstIdx);

  bool operator==(MultiDiEdge const &) const;
  bool operator<(MultiDiEdge const &) const;

  using AsConstTuple = std::tuple<Node, Node, std::size_t, std::size_t>;
  AsConstTuple as_tuple() const;

  std::string to_string() const;
public:
  Node src, dst;
  std::size_t srcIdx, dstIdx;
};
std::ostream &operator<<(std::ostream &, MultiDiEdge const &);

}
}

namespace std {
template <>
struct hash<::FlexFlow::utils::MultiDiEdge> {
  std::size_t operator()(::FlexFlow::utils::MultiDiEdge const &) const;
};
}


namespace FlexFlow {
namespace utils {

struct MultiDiEdgeQuery {
  tl::optional<std::unordered_set<Node>> srcs, dsts;
  tl::optional<std::unordered_set<std::size_t>> srcIdxs, dstIdxs;

  MultiDiEdgeQuery with_src_nodes(std::unordered_set<Node> const &) const;
  MultiDiEdgeQuery with_src_node(Node const &) const;
  MultiDiEdgeQuery with_dst_nodes(std::unordered_set<Node> const &) const;
  MultiDiEdgeQuery with_dst_node(Node const &) const;
  MultiDiEdgeQuery with_src_idxs(std::unordered_set<std::size_t> const &) const;
  MultiDiEdgeQuery with_src_idx(std::size_t) const;
  MultiDiEdgeQuery with_dst_idxs(std::unordered_set<std::size_t> const &) const;
  MultiDiEdgeQuery with_dst_idx(std::size_t) const;

  static MultiDiEdgeQuery all();
};

MultiDiEdgeQuery query_intersection(MultiDiEdgeQuery const &, MultiDiEdgeQuery const &);

struct IMultiDiGraphView : public IGraphView {
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
};

struct IMultiDiGraph : public IMultiDiGraphView, public IGraph {
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;
};

}
}

#endif 

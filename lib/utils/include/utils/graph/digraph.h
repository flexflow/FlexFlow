#ifndef _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H

#include "node.h"
#include "tl/optional.hpp"
#include <unordered_set>

namespace FlexFlow {

struct DirectedEdge {
public:
  DirectedEdge() = delete;
  DirectedEdge(Node src, Node dst);

  bool operator==(DirectedEdge const &) const;
  bool operator<(DirectedEdge const &) const;

  using AsConstTuple = std::tuple<Node, Node>;
  AsConstTuple as_tuple() const;
public:
  Node src, dst;
};
std::ostream &operator<<(std::ostream &, DirectedEdge const &);

}
}

namespace std {
template <>
struct hash<::FlexFlow::utils::DirectedEdge> {
  std::size_t operator()(::FlexFlow::utils::DirectedEdge const &) const;
};
}

namespace FlexFlow {
namespace utils {

struct DirectedEdgeQuery {
  DirectedEdgeQuery() = default;
  DirectedEdgeQuery(tl::optional<std::unordered_set<Node>> const &srcs, tl::optional<std::unordered_set<Node>> const &dsts);
  tl::optional<std::unordered_set<Node>> srcs = tl::nullopt, 
                                         dsts = tl::nullopt;
};

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &, DirectedEdgeQuery const &);

struct IDiGraphView : public IGraphView {
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
};


struct IDiGraph : public IDiGraphView, public IGraph {
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;
};

}

#endif

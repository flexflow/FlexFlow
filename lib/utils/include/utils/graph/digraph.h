#ifndef _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H

#include "node.h"
#include "tl/optional.hpp"
#include <unordered_set>
#include "utils/visitable.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct DirectedEdge {
public:
  DirectedEdge() = delete;
  DirectedEdge(Node src, Node dst);

  bool operator==(DirectedEdge const &) const;
  bool operator<(DirectedEdge const &) const;
public:
  Node src, dst;
};
std::ostream &operator<<(std::ostream &, DirectedEdge const &);

}

namespace std {
template <>
struct hash<::FlexFlow::DirectedEdge> {
  std::size_t operator()(::FlexFlow::DirectedEdge const &) const;
};
}

namespace FlexFlow {

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

VISITABLE_STRUCT(::FlexFlow::DirectedEdge, src, dst);

#endif

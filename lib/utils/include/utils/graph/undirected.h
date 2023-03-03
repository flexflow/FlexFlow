#ifndef _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H
#define _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H

#include "tl/optional.hpp"
#include <unordered_set>
#include "node.h"

namespace FlexFlow {

struct UndirectedEdge {
public:
  UndirectedEdge() = delete;
  UndirectedEdge(Node src, Node dst);

  bool operator==(UndirectedEdge const &) const;
  bool operator<(UndirectedEdge const &) const;
public:
  Node smaller, bigger;
};

}

namespace std {
template <>
struct hash<::FlexFlow::UndirectedEdge> {
  std::size_t operator()(::FlexFlow::UndirectedEdge const &) const;
};
}

namespace FlexFlow {

struct UndirectedEdgeQuery {
  UndirectedEdgeQuery(tl::optional<std::unordered_set<Node>> const &);

  tl::optional<std::unordered_set<Node>> nodes = tl::nullopt;
};

struct IUndirectedGraphView : public IGraphView {
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  virtual std::unordered_set<Edge> query_edges(UndirectedEdgeQuery const &) const = 0;
};

struct IUndirectedGraph : public IUndirectedGraphView, public IGraph {
  virtual void add_edge(UndirectedEdge const &) = 0;
  virtual void remove_edge(UndirectedEdge const &) = 0;
};

}

VISITABLE_STRUCT(::FlexFlow::UndirectedEdge, smaller, bigger);

#endif

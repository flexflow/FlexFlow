#ifndef _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H
#define _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H

#include "tl/optional.hpp"
#include <unordered_set>
#include "node.h"

namespace FlexFlow {
namespace utils {

struct UndirectedEdge {
public:
  UndirectedEdge() = delete;
  UndirectedEdge(Node src, Node dst);

  bool operator==(UndirectedEdge const &) const;
  bool operator<(UndirectedEdge const &) const;

  using AsConstTuple = std::tuple<Node, Node>;
  AsConstTuple as_tuple() const;
public:
  Node smaller, bigger;
};

}
}

namespace std {
template <>
struct hash<::FlexFlow::utils::UndirectedEdge> {
  std::size_t operator()(::FlexFlow::utils::UndirectedEdge const &) const;
};
}

namespace FlexFlow {
namespace utils {

struct UndirectedEdgeQuery {
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

}

#endif

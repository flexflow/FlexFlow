#ifndef _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H
#define _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_H

#include "tl/optional.hpp"
#include <unordered_set>
#include "node.h"

namespace FlexFlow {
namespace utils {
namespace graph {
namespace undirected {

struct Edge {
public:
  Edge() = delete;
  Edge(Node src, Node dst);

  bool operator==(Edge const &) const;
  bool operator<(Edge const &) const;

  using AsConstTuple = std::tuple<Node, Node>;
  AsConstTuple as_tuple() const;
public:
  Node smaller, bigger;
};

}
}
}
}

namespace std {
template <>
struct hash<::FlexFlow::utils::graph::undirected::Edge> {
  std::size_t operator()(::FlexFlow::utils::graph::undirected::Edge const &) const;
};
}

namespace FlexFlow {
namespace utils {
namespace graph {
namespace undirected {

struct EdgeQuery {
  tl::optional<std::unordered_set<Node>> nodes = tl::nullopt;
};

struct IUndirectedGraphView : public IGraphView {
  using Edge = undirected::Edge;
  using EdgeQuery = undirected::EdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
};

struct IUndirectedGraph : public IUndirectedGraphView, public IGraph {
  virtual void add_edge(Edge const &) = 0;
};

}

using IUndirectedGraph = undirected::IUndirectedGraph;
using IUndirectedGraphView = undirected::IUndirectedGraphView;

}
}
}

#endif

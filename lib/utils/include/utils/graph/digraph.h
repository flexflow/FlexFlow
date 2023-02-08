#ifndef _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H
#define _FLEXFLOW_UTILS_GRAPH_DIGRAPH_H

#include "node.h"
#include "tl/optional.hpp"
#include <unordered_set>

namespace FlexFlow {
namespace utils {
namespace graph {
namespace digraph {

struct Edge {
public:
  Edge() = delete;
  Edge(Node src, Node dst);

  bool operator==(Edge const &) const;
  bool operator<(Edge const &) const;

  using AsConstTuple = std::tuple<Node, Node>;
  AsConstTuple as_tuple() const;
public:
  Node src, dst;
};
std::ostream &operator<<(std::ostream &, Edge const &);

}
}
}
}

namespace std {
template <>
struct hash<::FlexFlow::utils::graph::digraph::Edge> {
  std::size_t operator()(::FlexFlow::utils::graph::digraph::Edge const &) const;
};
}

namespace FlexFlow {
namespace utils {
namespace graph {
namespace digraph {

struct EdgeQuery {
  EdgeQuery() = default;
  EdgeQuery(tl::optional<std::unordered_set<Node>> const &srcs, tl::optional<std::unordered_set<Node>> const &dsts);
  tl::optional<std::unordered_set<Node>> srcs = tl::nullopt, 
                                         dsts = tl::nullopt;
};

struct IDiGraphView : public IGraphView {
  using Edge = digraph::Edge;
  using EdgeQuery = digraph::EdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
};

struct IDiGraph : public IDiGraphView, public IGraph {
  virtual void add_edge(Edge const &) = 0;
};

}

using IDiGraph = digraph::IDiGraph;
using IDiGraphView = digraph::IDiGraphView;

}
}
}

#endif

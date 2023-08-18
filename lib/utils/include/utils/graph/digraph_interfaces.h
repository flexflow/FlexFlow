#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_INTERFACES_H

#include "utils/type_traits.h"
#include "node.h"

namespace FlexFlow {

struct DirectedEdge {
  Node src;
  Node dst;
};
FF_VISITABLE_STRUCT(DirectedEdge, src, dst);

struct DirectedEdgeQuery {
  query_set<Node> srcs;
  query_set<Node> dsts;

  static DirectedEdgeQuery all();
};
FF_VISITABLE_STRUCT(DirectedEdgeQuery, srcs, dsts);

DirectedEdgeQuery query_intersection(DirectedEdgeQuery const &,
                                     DirectedEdgeQuery const &);

struct IDiGraphView : public IGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  IDiGraphView(IDiGraphView const &) = delete;
  IDiGraphView &operator=(IDiGraphView const &) = delete;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual ~IDiGraphView() = default;

protected:
  IDiGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDiGraphView);

struct IDiGraph : public IDiGraphView, public IGraph {
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;
  virtual IDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDiGraph);


}

#endif

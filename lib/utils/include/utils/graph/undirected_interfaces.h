#ifndef _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_INTERFACES_H
#define _FLEXFLOW_UTILS_GRAPH_UNDIRECTED_INTERFACES_H

#include "utils/type_traits.h"
#include "undirected_edge.h"
#include "utils/uid.h"

namespace FlexFlow {

struct UndirectedEdgeQuery {
  query_set<Node> nodes;

  static UndirectedEdgeQuery all() {
    NOT_IMPLEMENTED();
  }
};
FF_VISITABLE_STRUCT(UndirectedEdgeQuery, nodes);

UndirectedEdgeQuery query_intersection(UndirectedEdgeQuery const &,
                                       UndirectedEdgeQuery const &);

struct IUndirectedGraphView : public IGraphView {
  using Edge = UndirectedEdge;
  using EdgeQuery = UndirectedEdgeQuery;

  IUndirectedGraphView(IUndirectedGraphView const &) = delete;
  IUndirectedGraphView &operator=(IUndirectedGraphView const &) = delete;

  virtual std::unordered_set<Edge>
      query_edges(UndirectedEdgeQuery const &) const = 0;
  virtual ~IUndirectedGraphView();

protected:
  IUndirectedGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUndirectedGraphView);

struct IUndirectedGraph : public IUndirectedGraphView, public IGraph {
  virtual void add_edge(UndirectedEdge const &) = 0;
  virtual void remove_edge(UndirectedEdge const &) = 0;

  virtual IUndirectedGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IUndirectedGraph);

}

#endif

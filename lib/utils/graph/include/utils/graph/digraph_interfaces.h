#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_DIGRAPH_INTERFACES
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_DIGRAPH_INTERFACES

#include "node.h"
#include "utils/type_traits.h"

namespace FlexFlow {

struct IDiGraphView : virtual public IGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  IDiGraphView() = default;

  IDiGraphView(IDiGraphView const &) = delete;
  IDiGraphView &operator=(IDiGraphView const &) = delete;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual ~IDiGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDiGraphView);

struct IDiGraph : virtual public IDiGraphView {
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;
  virtual IDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDiGraph);

} // namespace FlexFlow

#endif

#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_INTERFACES
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_INTERFACES

#include "multidiedge.h"
#include "node.h"
#include "query_set.h"
#include "utils/optional.h"
#include "utils/strong_typedef.h"
#include "utils/visitable.h"
#include "digraph_interfaces.h"

namespace FlexFlow {

struct IMultiDiGraphView : virtual public IDiGraphView {
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  std::unordered_set<DirectedEdge> query_edges(DirectedEdgeQuery const &) const override final;
  virtual ~IMultiDiGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IMultiDiGraphView);

struct IMultiDiGraph : virtual public IMultiDiGraphView {
  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &) = 0;
  virtual void remove_node_unsafe(Node const &) = 0;
  virtual NodePort add_node_port() = 0;
  virtual void add_node_port_unsafe(NodePort const &) = 0;
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;

  virtual std::unordered_set<Node>
      query_nodes(NodeQuery const &query) const override {
    return static_cast<IMultiDiGraphView const *>(this)->query_nodes(query);
  }

  virtual IMultiDiGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IMultiDiGraph);

} // namespace FlexFlow

#endif

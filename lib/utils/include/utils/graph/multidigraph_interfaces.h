#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_INTERFACES_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_INTERFACES_H

#include "multidiedge.h"
#include "node.h"
#include "query_set.h"
#include "utils/optional.h"
#include "utils/strong_typedef.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct MultiDiEdgeQuery {
  query_set<Node> srcs;
  query_set<Node> dsts;
  query_set<NodePort> srcIdxs;
  query_set<NodePort> dstIdxs;

  MultiDiEdgeQuery with_src_nodes(query_set<Node> const &) const;
  MultiDiEdgeQuery with_dst_nodes(query_set<Node> const &) const;
  MultiDiEdgeQuery with_src_idxs(query_set<NodePort> const &) const;
  MultiDiEdgeQuery with_dst_idxs(query_set<NodePort> const &) const;

  static MultiDiEdgeQuery all();
  static MultiDiEdgeQuery none();
};
FF_VISITABLE_STRUCT(MultiDiEdgeQuery, srcs, dsts, srcIdxs, dstIdxs);

MultiDiEdgeQuery query_intersection(MultiDiEdgeQuery const &,
                                    MultiDiEdgeQuery const &);
MultiDiEdgeQuery query_union(MultiDiEdgeQuery const &,
                             MultiDiEdgeQuery const &);

struct IMultiDiGraphView : public IGraphView {
  using Edge = MultiDiEdge;
  using EdgeQuery = MultiDiEdgeQuery;

  virtual std::unordered_set<Edge> query_edges(EdgeQuery const &) const = 0;
  virtual ~IMultiDiGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IMultiDiGraphView);

struct IMultiDiGraph : public IMultiDiGraphView, public IGraph {
  virtual NodePort add_node_port() = 0;
  virtual void add_node_port_unsafe(NodePort const &) = 0;
  virtual void add_edge(Edge const &) = 0;
  virtual void remove_edge(Edge const &) = 0;

  virtual std::unordered_set<Node>
      query_nodes(NodeQuery const &query) const override {
    return static_cast<IMultiDiGraphView const *>(this)->query_nodes(query);
  }

  virtual IMultiDiGraph *clone() const override = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IMultiDiGraph);

} // namespace FlexFlow

#endif

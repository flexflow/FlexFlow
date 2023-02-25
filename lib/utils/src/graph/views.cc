#include "utils/graph/views.h"
#include "utils/containers.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph.h"

namespace FlexFlow {
namespace utils {

FlippedView::FlippedView(IDiGraphView const &g)
  : g(g)
{ }

std::unordered_set<DirectedEdge> FlippedView::query_edges(DirectedEdgeQuery const &query) const {
  std::unordered_set<DirectedEdge> result = this->g.query_edges({query.dsts, query.srcs});
  return map_over_unordered_set<DirectedEdge, DirectedEdge>(flipped, result);
}

std::unordered_set<Node> FlippedView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(query);
}

DirectedEdge flipped(DirectedEdge const &e) {
  return {e.src, e.dst};  
}


FlippedView unsafe_view_as_flipped(IDiGraphView const &g) {
  return FlippedView(g);
}

DiSubgraphView::DiSubgraphView(IDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes)
  : g(g), subgraph_nodes(subgraph_nodes)
{ } 

std::unordered_set<DirectedEdge> DiSubgraphView::query_edges(DirectedEdgeQuery const &query) const {
  DirectedEdgeQuery subgraph_query = {this->subgraph_nodes, this->subgraph_nodes};
  return this->g.query_edges(query_intersection(query, subgraph_query));
}

std::unordered_set<Node> DiSubgraphView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(query_intersection(query, {this->subgraph_nodes}));
}

MultiDiSubgraphView::MultiDiSubgraphView(IMultiDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes)
  : g(g), subgraph_nodes(subgraph_nodes)
{ }

std::unordered_set<MultiDiEdge> MultiDiSubgraphView::query_edges(MultiDiEdgeQuery const &query) const {
  MultiDiEdgeQuery subgraph_query = MultiDiEdgeQuery::all().with_src_nodes(this->subgraph_nodes).with_dst_nodes(this->subgraph_nodes);
  return this->g.query_edges(query_intersection(query, subgraph_query));
}

DiSubgraphView unsafe_view_subgraph(IDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return DiSubgraphView(g, subgraph_nodes);
}

MultiDiSubgraphView unsafe_view_subgraph(IMultiDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return MultiDiSubgraphView(g, subgraph_nodes);
}

Node NodeSource::fresh_node() {
  Node result(this->next_node_idx);
  this->next_node_idx++;
  return result;
}

JoinedNodeView::JoinedNodeView(IGraphView const &lhs, IGraphView const &rhs) {
  for (Node const &n : get_nodes(lhs)) {
    this->mapping.equate({n, true}, this->node_source.fresh_node());
  }
  for (Node const &n : get_nodes(rhs)) {
    this->mapping.equate({n, false}, this->node_source.fresh_node());
  }
}

std::unordered_set<Node> JoinedNodeView::query_nodes(NodeQuery const &query) const {
  std::unordered_set<Node> result;
  for (auto const &kv : this->mapping) {
    if (!query.nodes.has_value() || contains(query.nodes.value(), kv.second)) {
      result.insert(kv.second);
    }
  }
  return result;
}

std::pair<std::unordered_set<Node>, std::unordered_set<Node>> JoinedNodeView::trace_nodes(std::unordered_set<Node> const &nodes) const {
  std::unordered_set<Node> left_nodes, right_nodes;

  for (Node const &n : nodes) {
    JoinNodeKey k = this->at_node(n);
    if (k.is_left) {
      left_nodes.insert(k.node);
    } else {
      right_nodes.insert(k.node);
    }
  }

  return {left_nodes, right_nodes};
}

Node JoinedNodeView::at_join_key(JoinNodeKey const &k) const {
  return this->mapping.at_l(k);
}

JoinNodeKey JoinedNodeView::at_node(Node const &n) const {
  return this->mapping.at_r(n);
}

JoinedUndirectedGraphView::JoinedUndirectedGraphView(IUndirectedGraphView const &lhs, IUndirectedGraphView const &rhs)
  : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs)
{ }

std::unordered_set<Node> JoinedUndirectedGraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<UndirectedEdge> JoinedUndirectedGraphView::query_edges(UndirectedEdgeQuery const &query) const {
  std::unordered_set<Node> nodes = query.nodes.value_or(get_nodes(*this));
  std::unordered_set<Node> left_nodes, right_nodes;
  for (Node const &n : nodes) {
    JoinNodeKey k = this->joined_nodes.at_node(n);
    if (k.is_left) {
      left_nodes.insert(k.node);
    } else {
      right_nodes.insert(k.node);
    }
  }
  UndirectedEdgeQuery left_query(left_nodes);
  UndirectedEdgeQuery right_query(right_nodes);

  std::unordered_set<UndirectedEdge> result;
  for (UndirectedEdge const &e : this->lhs.query_edges(left_query)) {
    result.insert(this->fix_lhs_edge(e)); 
  }
  for (UndirectedEdge const &e : this->rhs.query_edges(right_query)) {
    result.insert(this->fix_rhs_edge(e));
  }

  return result;
}

UndirectedEdge JoinedUndirectedGraphView::fix_lhs_edge(UndirectedEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.smaller, true}),
    this->joined_nodes.at_join_key({e.bigger, true})
  };
}

UndirectedEdge JoinedUndirectedGraphView::fix_rhs_edge(UndirectedEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.smaller, false}),
    this->joined_nodes.at_join_key({e.bigger, false})
  };
}

JoinedDigraphView::JoinedDigraphView(IDiGraphView const &lhs, IDiGraphView const &rhs) 
  : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs)
{ }

std::unordered_set<Node> JoinedDigraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<DirectedEdge> JoinedDigraphView::query_edges(DirectedEdgeQuery const &query) const {
  std::unordered_set<Node> srcs = query.srcs.value_or(get_nodes(*this));
  std::unordered_set<Node> dsts = query.dsts.value_or(get_nodes(*this));
  auto traced_srcs = this->joined_nodes.trace_nodes(srcs);
  auto traced_dsts = this->joined_nodes.trace_nodes(dsts);
  DirectedEdgeQuery left_query(traced_srcs.first, traced_dsts.first);
  DirectedEdgeQuery right_query(traced_srcs.second, traced_dsts.second);

  std::unordered_set<DirectedEdge> result;
  for (DirectedEdge const &e : this->lhs.query_edges(left_query)) {
    result.insert(this->fix_lhs_edge(e)); 
  }
  for (DirectedEdge const &e : this->rhs.query_edges(right_query)) {
    result.insert(this->fix_rhs_edge(e));
  }

  return result;
}

DirectedEdge JoinedDigraphView::fix_lhs_edge(DirectedEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.src, true}),
    this->joined_nodes.at_join_key({e.dst, true})
  };
}

DirectedEdge JoinedDigraphView::fix_rhs_edge(DirectedEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.src, false}),
    this->joined_nodes.at_join_key({e.dst, false})
  };
}

JoinedMultiDigraphView::JoinedMultiDigraphView(IMultiDiGraphView const &lhs, IMultiDiGraphView const &rhs)
  : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs)
{ }

std::unordered_set<Node> JoinedMultiDigraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<MultiDiEdge> JoinedMultiDigraphView::query_edges(MultiDiEdgeQuery const &query) const {
  std::unordered_set<Node> srcs = query.srcs.value_or(get_nodes(*this));
  std::unordered_set<Node> dsts = query.dsts.value_or(get_nodes(*this));

  auto traced_srcs = this->joined_nodes.trace_nodes(srcs);
  auto traced_dsts = this->joined_nodes.trace_nodes(dsts);
  MultiDiEdgeQuery left_query(traced_srcs.first, traced_dsts.first, query.srcIdxs, query.dstIdxs);
  MultiDiEdgeQuery right_query(traced_srcs.second, traced_dsts.second, query.srcIdxs, query.dstIdxs);

  std::unordered_set<MultiDiEdge> result;
  for (MultiDiEdge const &e : this->lhs.query_edges(left_query)) {
    result.insert(this->fix_lhs_edge(e));
  }
  for (MultiDiEdge const &e : this->rhs.query_edges(right_query)) {
    result.insert(this->fix_rhs_edge(e));
  }

  return result;
}

MultiDiEdge JoinedMultiDigraphView::fix_lhs_edge(MultiDiEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.src, true}),
    this->joined_nodes.at_join_key({e.dst, true}),
    e.srcIdx,
    e.dstIdx
  };
}

MultiDiEdge JoinedMultiDigraphView::fix_rhs_edge(MultiDiEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.src, false}),
    this->joined_nodes.at_join_key({e.dst, false}),
    e.srcIdx,
    e.dstIdx
  };
}


}
}

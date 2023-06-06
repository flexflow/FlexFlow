#include "utils/graph/views.h"
#include "utils/containers.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph.h"
#include "utils/disjoint_set.h"

namespace FlexFlow {

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

DiGraphView unsafe_view_as_flipped(IDiGraphView const &g) {
  return DiGraphView::create<FlippedView>(g);
}

UndirectedSubgraphView::UndirectedSubgraphView(maybe_owned_ref<IUndirectedGraphView const> g, std::unordered_set<Node> const &subgraph_nodes)
  : g(g), subgraph_nodes(subgraph_nodes)
{ }

std::unordered_set<UndirectedEdge> UndirectedSubgraphView::query_edges(UndirectedEdgeQuery const &query) const {
  UndirectedEdgeQuery subgraph_query = { this->subgraph_nodes };
  return this->g.get().query_edges(query_intersection(query, subgraph_query));
}

std::unordered_set<Node> UndirectedSubgraphView::query_nodes(NodeQuery const &query) const {
  return this->g.get().query_nodes(query_intersection(query, {this->subgraph_nodes}));
}

DiSubgraphView::DiSubgraphView(maybe_owned_ref<IDiGraphView const> g, std::unordered_set<Node> const &subgraph_nodes)
  : g(std::move(g)), subgraph_nodes(subgraph_nodes)
{ } 

std::unordered_set<DirectedEdge> DiSubgraphView::query_edges(DirectedEdgeQuery const &query) const {
  DirectedEdgeQuery subgraph_query = {this->subgraph_nodes, this->subgraph_nodes};
  return this->g.get().query_edges(query_intersection(query, subgraph_query));
}

std::unordered_set<Node> DiSubgraphView::query_nodes(NodeQuery const &query) const {
  return this->g.get().query_nodes(query_intersection(query, {this->subgraph_nodes}));
}

MultiDiSubgraphView::MultiDiSubgraphView(maybe_owned_ref<IMultiDiGraphView const> g, std::unordered_set<Node> const &subgraph_nodes)
  : g(g), subgraph_nodes(subgraph_nodes)
{ }

std::unordered_set<MultiDiEdge> MultiDiSubgraphView::query_edges(MultiDiEdgeQuery const &query) const {
  MultiDiEdgeQuery subgraph_query = MultiDiEdgeQuery::all().with_src_nodes(this->subgraph_nodes).with_dst_nodes(this->subgraph_nodes);
  return this->g.get().query_edges(query_intersection(query, subgraph_query));
}

UndirectedGraphView unsafe_view_subgraph(UndirectedGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return UndirectedGraphView::create<UndirectedSubgraphView>(g.unsafe(), subgraph_nodes);
}

UndirectedGraphView view_subgraph(UndirectedGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return UndirectedGraphView::create<UndirectedSubgraphView>(g, subgraph_nodes);
}

DiGraphView unsafe_view_subgraph(DiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return DiGraphView::create<DiSubgraphView>(g.unsafe(), subgraph_nodes);
}

DiGraphView view_subgraph(DiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return DiGraphView::create<DiSubgraphView>(g, subgraph_nodes);
}

MultiDiGraphView unsafe_view_subgraph(MultiDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return MultiDiGraphView::create<MultiDiSubgraphView>(g.unsafe(), subgraph_nodes);
}

MultiDiGraphView view_subgraph(MultiDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes) {
  return MultiDiGraphView::create<MultiDiSubgraphView>(g, subgraph_nodes);
}

Node NodeSource::fresh_node() {
  Node result(this->next_node_idx);
  this->next_node_idx++;
  return result;
}

JoinedNodeView::JoinedNodeView(IGraphView const &lhs, IGraphView const &rhs) {
  for (Node const &n : get_nodes(GraphView::unsafe(lhs))) {
    this->mapping.equate({n, LRDirection::LEFT}, this->node_source.fresh_node());
  }
  for (Node const &n : get_nodes(GraphView::unsafe(rhs))) {
    this->mapping.equate({n, LRDirection::RIGHT}, this->node_source.fresh_node());
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
    if (k.direction == LRDirection::LEFT) {
      left_nodes.insert(k.node);
    } else {
      assert (k.direction == LRDirection::RIGHT);
      right_nodes.insert(k.node);
    }
  }

  return {left_nodes, right_nodes};
}

JoinNodeKey::JoinNodeKey(Node const & node, LRDirection direction)
:node(node), direction(direction){}

bool JoinNodeKey::operator==(JoinNodeKey const & jnk) const {
  return node== jnk.node && direction == jnk.direction;
}

Node JoinedNodeView::at_join_key(JoinNodeKey const &k) const {
  return this->mapping.at_l(k);
}

JoinNodeKey JoinedNodeView::at_node(Node const &n) const {
  return this->mapping.at_r(n);
}

JoinedUndirectedGraphView::JoinedUndirectedGraphView(maybe_owned_ref<IUndirectedGraphView const> lhs, maybe_owned_ref<IUndirectedGraphView const> rhs)
  : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs)
{ }

std::unordered_set<Node> JoinedUndirectedGraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<UndirectedEdge> JoinedUndirectedGraphView::query_edges(UndirectedEdgeQuery const &query) const {
  std::unordered_set<Node> nodes = query.nodes.value_or(get_nodes(unsafe(*this)));
  std::unordered_set<Node> left_nodes, right_nodes;
  for (Node const &n : nodes) {
    JoinNodeKey k = this->joined_nodes.at_node(n);
    if (k.direction == LRDirection::LEFT) {
      left_nodes.insert(k.node);
    } else {
      assert (k.direction == LRDirection::RIGHT);
      right_nodes.insert(k.node);
    }
  }
  UndirectedEdgeQuery left_query(left_nodes);
  UndirectedEdgeQuery right_query(right_nodes);

  std::unordered_set<UndirectedEdge> result;
  for (UndirectedEdge const &e : this->lhs.get().query_edges(left_query)) {
    result.insert(this->fix_lhs_edge(e)); 
  }
  for (UndirectedEdge const &e : this->rhs.get().query_edges(right_query)) {
    result.insert(this->fix_rhs_edge(e));
  }

  return result;
}

UndirectedEdge JoinedUndirectedGraphView::fix_lhs_edge(UndirectedEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.smaller, LRDirection::LEFT}),
    this->joined_nodes.at_join_key({e.bigger, LRDirection::LEFT})
  };
}

UndirectedEdge JoinedUndirectedGraphView::fix_rhs_edge(UndirectedEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.smaller, LRDirection::RIGHT}),
    this->joined_nodes.at_join_key({e.bigger, LRDirection::RIGHT})
  };
}

JoinedDigraphView::JoinedDigraphView(maybe_owned_ref<IDiGraphView const> lhs, maybe_owned_ref<IDiGraphView const> rhs) 
  : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs)
{ }

std::unordered_set<Node> JoinedDigraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<DirectedEdge> JoinedDigraphView::query_edges(DirectedEdgeQuery const &query) const {
  std::unordered_set<Node> srcs = query.srcs.value_or(get_nodes(unsafe(*this)));
  std::unordered_set<Node> dsts = query.dsts.value_or(get_nodes(unsafe(*this)));
  auto traced_srcs = this->joined_nodes.trace_nodes(srcs);
  auto traced_dsts = this->joined_nodes.trace_nodes(dsts);
  DirectedEdgeQuery left_query(traced_srcs.first, traced_dsts.first);
  DirectedEdgeQuery right_query(traced_srcs.second, traced_dsts.second);

  std::unordered_set<DirectedEdge> result;
  for (DirectedEdge const &e : this->lhs.get().query_edges(left_query)) {
    result.insert(this->fix_lhs_edge(e)); 
  }
  for (DirectedEdge const &e : this->rhs.get().query_edges(right_query)) {
    result.insert(this->fix_rhs_edge(e));
  }

  return result;
}

DirectedEdge JoinedDigraphView::fix_lhs_edge(DirectedEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.src, LRDirection::LEFT}),
    this->joined_nodes.at_join_key({e.dst, LRDirection::LEFT})
  };
}

DirectedEdge JoinedDigraphView::fix_rhs_edge(DirectedEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.src, LRDirection::RIGHT}),
    this->joined_nodes.at_join_key({e.dst, LRDirection::RIGHT})
  };
}

JoinedMultiDigraphView::JoinedMultiDigraphView(maybe_owned_ref<IMultiDiGraphView const> lhs, maybe_owned_ref<IMultiDiGraphView const> rhs)
  : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs)
{ }

std::unordered_set<Node> JoinedMultiDigraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<MultiDiEdge> JoinedMultiDigraphView::query_edges(MultiDiEdgeQuery const &query) const {
  std::unordered_set<Node> srcs = query.srcs.value_or(get_nodes(unsafe(*this)));
  std::unordered_set<Node> dsts = query.dsts.value_or(get_nodes(unsafe(*this)));

  auto traced_srcs = this->joined_nodes.trace_nodes(srcs);
  auto traced_dsts = this->joined_nodes.trace_nodes(dsts);
  MultiDiEdgeQuery left_query(traced_srcs.first, traced_dsts.first, query.srcIdxs, query.dstIdxs);
  MultiDiEdgeQuery right_query(traced_srcs.second, traced_dsts.second, query.srcIdxs, query.dstIdxs);

  std::unordered_set<MultiDiEdge> result;
  for (MultiDiEdge const &e : this->lhs.get().query_edges(left_query)) {
    result.insert(this->fix_lhs_edge(e));
  }
  for (MultiDiEdge const &e : this->rhs.get().query_edges(right_query)) {
    result.insert(this->fix_rhs_edge(e));
  }

  return result;
}

MultiDiEdge JoinedMultiDigraphView::fix_lhs_edge(MultiDiEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.src, LRDirection::LEFT}),
    this->joined_nodes.at_join_key({e.dst, LRDirection::LEFT}),
    e.srcIdx,
    e.dstIdx
  };
}

MultiDiEdge JoinedMultiDigraphView::fix_rhs_edge(MultiDiEdge const &e) const {
  return {
    this->joined_nodes.at_join_key({e.src, LRDirection::RIGHT}),
    this->joined_nodes.at_join_key({e.dst, LRDirection::RIGHT}),
    e.srcIdx,
    e.dstIdx
  };
}

/* SingleSourceNodeView::SingleSourceNodeView(IDiGraphView const &g) */
/*   : g(g) */
/* { */
/*   std::unordered_set<Node> sources = get_sources(g); */
/*   if (sources.size() == 1) { */
/*     this->singleton_src = tl::nullopt; */
/*     this->joined_view = tl::nullopt; */
/*   } else { */
/*     AdjacencyDiGraph singleton_src_g; */
/*     Node new_src_node = singleton_src_g.add_node(); */
/*     this->joined_view = JoinedNodeView(g, singleton_src_g); */
/*     std::unordered_set<DirectedEdge> new_edges; */
/*     JoinedNodeView const &joined_nodes_view = this->joined_view.joined_nodes_view(); */
/*     for (Node const &src_node : sources) { */
/*       Node joined_src_node = joined_nodes_view.at_join_key({src_node, LRDirection::LEFT}); */
/*       Node joined_new_src_node = joined_nodes_view.at_join_key({new_src_node, LRDirection::RIGHT}); */
/*       new_edges.insert({joined_new_src_node, joined_src_node}); */
/*     } */
/*     this->added_edges_view = unsafe_view_with_added_edges(this->joined_view.value(), new_edges); */
/*     this->singleton_src = singleton_src_g; */
/*   } */
/* } */

/* std::unordered_set<Node> SingleSourceNodeView::query_nodes(NodeQuery const &q) const { */
/*   if (this->joined_nodes.has_value()) { */
/*     return this->joined_nodes.value().query_nodes(q); */
/*   } else { */
/*     return this->g.query_nodes(q); */
/*   } */
/* } */

/* std::unordered_set<DirectedEdge> SingleSourceNodeView::query_edges(DirectedEdgeQuery const &q) const { */
/*   if (this->joined_view != nullptr) { */
/*     this->joined_view->query_edges(q); */
/*   } else { */
/*     return this->g.query_edges(q); */
/*   } */
/* } */

/* NodePermutationView::NodePermutationView(IDiGraphView const &g, Node const &from, Node const &to) */
/*   : g(g), from(from), to(to) */
/* { } */

/* std::unordered_set<Node> NodePermutationView::query_nodes(NodeQuery const &q) const { */
/*   NodeQuery qq; */
/*   for (Node const &n : q.nodes) { */
/*     if (n == this->from) { */
/*       qq.insert(this->to); */
/*     } else { */
/*       qq.insert(n); */
/*     } */
/*   } */
/*   return this->g.query_nodes(qq); */
/* } */

/* std::unordered_set<DirectedEdge> NodePermutationView::query_edges(DirectedEdgeQuery const &q) { */ 
/*   DirectedEdgeQuery qq = {std::unordered_set{}, std::unordered_set{}}; */
/*   for (Node const &src : q.srcs) { */
/*     qq.srcs.insert(src); */
/*     if (src == this->to) { */
/*       qq.srcs.insert(this->from); */
/*     } */
/*   } */
/*   for (Node const &dst : q.dsts) { */
/*     qq.dsts.insert(dst); */
/*     if (dsts = this->to) { */
/*       qq.dsts.insert(this->from); */
/*     } */
/*   } */
/*   std::unordered_set<DirectedEdge> edges = this->g.query_edges(qq); */
/*   return map_over_unordered_set(edges, [](DirectedEdge const &e) { return this->fix_edge(e); }); */
/* } */

/* DirectedEdge NodePermutationView::fix_edge(DirectedEdge const &e) { */
/*   DirectedEdge result = e; */
/*   if (result.src == this->from) { */
/*     result.src = this->to; */
/*   } */
/*   if (result.dst = this->from) { */
/*     result.dst = this->to; */
/*   } */
/* } */

/* std::unordered_map<Node, Node> flatten_contraction(std::unordered_map<Node, Node> const &m) { */
/*   disjoint_set<Node> unionfind; */
/*   for (auto const &kv : m) { */
/*     unionfind.m_union(kv.first, kv.second); */
/*   } */
/*   std::unordered_map<Node, Node> result.add_edge; */
/*   for (auto const &kv : m) { */
/*     result[kv.first] = unionfind.find(kv.first); */
/*   } */
/*   return result; */
/* } */

}

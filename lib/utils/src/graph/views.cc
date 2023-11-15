#include "utils/graph/views.h"
#include "utils/containers.h"
#include "utils/disjoint_set.h"
#include "utils/exception.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph.h"
#include "utils/graph/digraph_interfaces.h"
#include "utils/graph/query_set.h"
#include "utils/graph/undirected.h"
#include <unordered_set>

namespace FlexFlow {

FlippedView::FlippedView(DiGraphView const &g) : g(g) {}

std::unordered_set<DirectedEdge>
    FlippedView::query_edges(DirectedEdgeQuery const &query) const {
  std::unordered_set<DirectedEdge> result =
      this->g.query_edges({query.dsts, query.srcs});
  return transform(result, [](DirectedEdge const &e) { return flipped(e); });
}

std::unordered_set<Node>
    FlippedView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(query);
}

FlippedView *FlippedView::clone() const {
  return new FlippedView(g);
}

std::unordered_set<DirectedEdge>
    ContractNodeView::query_edges(DirectedEdgeQuery const &q) const {
  return g.query_edges(q);
}

std::unordered_set<Node>
    ContractNodeView::query_nodes(NodeQuery const &q) const {
  return g.query_nodes(q);
}

ContractNodeView *ContractNodeView::clone() const {
  return new ContractNodeView(g, from, to);
}

DirectedEdge flipped(DirectedEdge const &e) {
  return {e.src, e.dst};
}

UndirectedSubgraphView::UndirectedSubgraphView(
    UndirectedGraphView const &g,
    std::unordered_set<Node> const &subgraph_nodes)
    : g(g), subgraph_nodes(subgraph_nodes) {}

UndirectedSubgraphView *UndirectedSubgraphView::clone() const {
  return new UndirectedSubgraphView(g, subgraph_nodes);
}

std::unordered_set<UndirectedEdge> UndirectedSubgraphView::query_edges(
    UndirectedEdgeQuery const &query) const {
  UndirectedEdgeQuery subgraph_query = {this->subgraph_nodes};
  return this->g.query_edges(query_intersection(query, subgraph_query));
}

std::unordered_set<Node>
    UndirectedSubgraphView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(query_intersection(query, {this->subgraph_nodes}));
}

DiSubgraphView::DiSubgraphView(DiGraphView const &g,
                               std::unordered_set<Node> const &subgraph_nodes)
    : g(g), subgraph_nodes(subgraph_nodes) {}

std::unordered_set<DirectedEdge>
    DiSubgraphView::query_edges(DirectedEdgeQuery const &query) const {
  DirectedEdgeQuery subgraph_query = {this->subgraph_nodes,
                                      this->subgraph_nodes};
  return this->g.query_edges(query_intersection(query, subgraph_query));
}

std::unordered_set<Node>
    DiSubgraphView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(query_intersection(query, {this->subgraph_nodes}));
}

DiSubgraphView *DiSubgraphView::clone() const {
  return new DiSubgraphView(g, subgraph_nodes);
}

MultiDiSubgraphView::MultiDiSubgraphView(
    MultiDiGraphView const &g, std::unordered_set<Node> const &subgraph_nodes)
    : g(g), subgraph_nodes(subgraph_nodes) {}

std::unordered_set<MultiDiEdge>
    MultiDiSubgraphView::query_edges(MultiDiEdgeQuery const &query) const {
  MultiDiEdgeQuery subgraph_query = MultiDiEdgeQuery::all()
                                        .with_src_nodes(this->subgraph_nodes)
                                        .with_dst_nodes(this->subgraph_nodes);
  return this->g.query_edges(query_intersection(query, subgraph_query));
}

std::unordered_set<Node>
    MultiDiSubgraphView::query_nodes(NodeQuery const &query) const {
  return this->g.query_nodes(query_intersection(query, {this->subgraph_nodes}));
}

UndirectedGraphView
    view_subgraph(UndirectedGraphView const &g,
                  std::unordered_set<Node> const &subgraph_nodes) {
  return UndirectedGraphView::create<UndirectedSubgraphView>(g, subgraph_nodes);
}

DiGraphView view_subgraph(DiGraphView const &g,
                          std::unordered_set<Node> const &subgraph_nodes) {
  return DiGraphView::create<DiSubgraphView>(g, subgraph_nodes);
}

MultiDiGraphView view_subgraph(MultiDiGraphView const &g,
                               std::unordered_set<Node> const &subgraph_nodes) {
  return MultiDiGraphView::create<MultiDiSubgraphView>(g, subgraph_nodes);
}

Node NodeSource::fresh_node() {
  Node result(this->next_node_idx);
  this->next_node_idx++;
  return result;
}

JoinedNodeView::JoinedNodeView(GraphView const &lhs, GraphView const &rhs) {
  for (Node const &n : get_nodes(lhs)) {
    this->mapping.equate({n, LRDirection::LEFT},
                         this->node_source.fresh_node());
  }
  for (Node const &n : get_nodes(rhs)) {
    this->mapping.equate({n, LRDirection::RIGHT},
                         this->node_source.fresh_node());
  }
}

std::unordered_set<Node>
    JoinedNodeView::query_nodes(NodeQuery const &query) const {
  // TODO @lockshaw this is going to be reimplemented in 984, so don't bother
  // fixing it for now
  NOT_IMPLEMENTED();
}

std::pair<std::unordered_set<Node>, std::unordered_set<Node>>
    JoinedNodeView::trace_nodes(std::unordered_set<Node> const &nodes) const {
  std::unordered_set<Node> left_nodes, right_nodes;

  for (Node const &n : nodes) {
    JoinNodeKey k = this->at_node(n);
    if (k.direction == LRDirection::LEFT) {
      left_nodes.insert(k.node);
    } else {
      assert(k.direction == LRDirection::RIGHT);
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

JoinedUndirectedGraphView::JoinedUndirectedGraphView(
    UndirectedGraphView const &lhs, UndirectedGraphView const &rhs)
    : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs) {}

std::unordered_set<Node>
    JoinedUndirectedGraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<UndirectedEdge> JoinedUndirectedGraphView::query_edges(
    UndirectedEdgeQuery const &query) const {
  std::unordered_set<Node> nodes = this->query_nodes({query.nodes});
  std::unordered_set<Node> left_nodes, right_nodes;
  for (Node const &n : nodes) {
    JoinNodeKey k = this->joined_nodes.at_node(n);
    if (k.direction == LRDirection::LEFT) {
      left_nodes.insert(k.node);
    } else {
      assert(k.direction == LRDirection::RIGHT);
      right_nodes.insert(k.node);
    }
  }

  std::unordered_set<UndirectedEdge> result;
  for (UndirectedEdge const &e : this->lhs.query_edges({left_nodes})) {
    result.insert(this->fix_lhs_edge(e));
  }
  for (UndirectedEdge const &e : this->rhs.query_edges({right_nodes})) {
    result.insert(this->fix_rhs_edge(e));
  }

  return result;
}

UndirectedEdge
    JoinedUndirectedGraphView::fix_lhs_edge(UndirectedEdge const &e) const {
  return {this->joined_nodes.at_join_key({e.smaller, LRDirection::LEFT}),
          this->joined_nodes.at_join_key({e.bigger, LRDirection::LEFT})};
}

UndirectedEdge
    JoinedUndirectedGraphView::fix_rhs_edge(UndirectedEdge const &e) const {
  return {this->joined_nodes.at_join_key({e.smaller, LRDirection::RIGHT}),
          this->joined_nodes.at_join_key({e.bigger, LRDirection::RIGHT})};
}

JoinedDigraphView::JoinedDigraphView(DiGraphView const &lhs,
                                     DiGraphView const &rhs)
    : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs) {}

JoinedDigraphView *JoinedDigraphView::clone() const {
  return new JoinedDigraphView(lhs, rhs);
}

std::unordered_set<Node>
    JoinedDigraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<DirectedEdge>
    JoinedDigraphView::query_edges(DirectedEdgeQuery const &query) const {

  std::unordered_set<Node> srcs = this->query_nodes(query.srcs);
  std::unordered_set<Node> dsts = this->query_nodes(query.dsts);
  auto traced_srcs = this->joined_nodes.trace_nodes(srcs);
  auto traced_dsts = this->joined_nodes.trace_nodes(dsts);
  DirectedEdgeQuery left_query = {traced_srcs.first, traced_dsts.first};
  DirectedEdgeQuery right_query = {traced_srcs.second, traced_dsts.second};

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
  return {this->joined_nodes.at_join_key({e.src, LRDirection::LEFT}),
          this->joined_nodes.at_join_key({e.dst, LRDirection::LEFT})};
}

DirectedEdge JoinedDigraphView::fix_rhs_edge(DirectedEdge const &e) const {
  return {this->joined_nodes.at_join_key({e.src, LRDirection::RIGHT}),
          this->joined_nodes.at_join_key({e.dst, LRDirection::RIGHT})};
}

JoinedMultiDigraphView::JoinedMultiDigraphView(MultiDiGraphView const &lhs,
                                               MultiDiGraphView const &rhs)
    : lhs(lhs), rhs(rhs), joined_nodes(lhs, rhs) {}

std::unordered_set<Node>
    JoinedMultiDigraphView::query_nodes(NodeQuery const &query) const {
  return this->joined_nodes.query_nodes(query);
}

std::unordered_set<MultiDiEdge>
    JoinedMultiDigraphView::query_edges(MultiDiEdgeQuery const &query) const {
  std::unordered_set<Node> srcs = this->query_nodes(query.srcs);
  std::unordered_set<Node> dsts = this->query_nodes(query.dsts);

  auto traced_srcs = this->joined_nodes.trace_nodes(srcs);
  auto traced_dsts = this->joined_nodes.trace_nodes(dsts);
  MultiDiEdgeQuery left_query = {
      traced_srcs.first, traced_dsts.first, query.srcIdxs, query.dstIdxs};
  MultiDiEdgeQuery right_query = {
      traced_srcs.second, traced_dsts.second, query.srcIdxs, query.dstIdxs};

  return set_union(
      transform(this->lhs.query_edges(left_query),
                [&](MultiDiEdge const &e) { return this->fix_lhs_edge(e); }),
      transform(this->rhs.query_edges(right_query),
                [&](MultiDiEdge const &e) { return this->fix_rhs_edge(e); }));
}

JoinedMultiDigraphView *JoinedMultiDigraphView::clone() const {
  return new JoinedMultiDigraphView(lhs, rhs);
}

MultiDiEdge JoinedMultiDigraphView::fix_lhs_edge(MultiDiEdge const &e) const {
  return {this->joined_nodes.at_join_key({e.dst, LRDirection::LEFT}),
          e.dst_idx,
          this->joined_nodes.at_join_key({e.src, LRDirection::LEFT}),
          e.src_idx};
}

MultiDiEdge JoinedMultiDigraphView::fix_rhs_edge(MultiDiEdge const &e) const {
  return {this->joined_nodes.at_join_key({e.dst, LRDirection::RIGHT}),
          e.dst_idx,
          this->joined_nodes.at_join_key({e.src, LRDirection::RIGHT}),
          e.src_idx};
}

UndirectedEdge to_undirected_edge(DirectedEdge const &e) {
  return {e.src, e.dst};
}

std::unordered_set<UndirectedEdge> to_undirected_edges(
    std::unordered_set<DirectedEdge> const &directed_edges) {
  return transform(directed_edges,
                   [](DirectedEdge const &e) { return to_undirected_edge(e); });
}

UndirectedEdge to_undirected_edge(MultiDiEdge const &e) {
  return to_undirected_edge(to_directed_edge(e));
}

std::unordered_set<UndirectedEdge>
    to_undirected_edges(std::unordered_set<MultiDiEdge> const &multidi_edges) {
  return to_undirected_edges(to_directed_edges(multidi_edges));
}

std::unordered_set<DirectedEdge> to_directed_edges(UndirectedEdge const &e) {
  return {{e.smaller, e.bigger}, {e.bigger, e.smaller}};
}

std::unordered_set<DirectedEdge> to_directed_edges(
    std::unordered_set<UndirectedEdge> const &undirected_edges) {
  return flatmap_v2<DirectedEdge>(undirected_edges, to_directed_edges);
}

DirectedEdge to_directed_edge(MultiDiEdge const &e) {
  return {e.src, e.dst};
}

std::unordered_set<DirectedEdge>
    to_directed_edges(std::unordered_set<MultiDiEdge> const &multidi_edges) {
  return transform(multidi_edges, to_directed_edge);
}

ViewDiGraphAsUndirectedGraph::ViewDiGraphAsUndirectedGraph(DiGraphView const &g)
    : g(g) {}

std::unordered_set<UndirectedEdge> ViewDiGraphAsUndirectedGraph::query_edges(
    UndirectedEdgeQuery const &undirected_query) const {
  DirectedEdgeQuery q1{undirected_query.nodes, query_set<Node>::matchall()};
  DirectedEdgeQuery q2{query_set<Node>::matchall(), undirected_query.nodes};
  return to_undirected_edges(
      set_union(this->g.query_edges(q1), this->g.query_edges(q2)));
}

std::unordered_set<Node> ViewDiGraphAsUndirectedGraph::query_nodes(
    NodeQuery const &node_query) const {
  return this->g.query_nodes(node_query);
}

ViewDiGraphAsUndirectedGraph *ViewDiGraphAsUndirectedGraph::clone() const {
  return new ViewDiGraphAsUndirectedGraph(g);
}

ViewUndirectedGraphAsDiGraph::ViewUndirectedGraphAsDiGraph(
    UndirectedGraphView const &g)
    : g(g) {}

ViewUndirectedGraphAsDiGraph *ViewUndirectedGraphAsDiGraph::clone() const {
  return new ViewUndirectedGraphAsDiGraph(g);
}

std::unordered_set<DirectedEdge> ViewUndirectedGraphAsDiGraph::query_edges(
    DirectedEdgeQuery const &q) const {
  std::unordered_set<UndirectedEdge> undirected_edges =
      intersection(g.query_edges({q.srcs}), g.query_edges({q.dsts}));
  std::unordered_set<DirectedEdge> directed_edges =
      flatmap(undirected_edges,
              [](UndirectedEdge const &e) { return to_directed_edges(e); });
  return filter(directed_edges,
                [&](DirectedEdge const &e) { return matches_edge(q, e); });
}

std::unordered_set<Node>
    ViewUndirectedGraphAsDiGraph::query_nodes(NodeQuery const &q) const {
  return g.query_nodes(q);
}

ViewDiGraphAsMultiDiGraph::ViewDiGraphAsMultiDiGraph(DiGraphView const &g)
    : g(g) {}

std::unordered_set<MultiDiEdge> ViewDiGraphAsMultiDiGraph::query_edges(
    MultiDiEdgeQuery const &multidi_query) const {
  DirectedEdgeQuery directed_query{multidi_query.srcs, multidi_query.dsts};

  std::unordered_set<DirectedEdge> const directed_edges =
      this->g.query_edges(directed_query);

  return transform(directed_edges, [](DirectedEdge const &e) {
    return MultiDiEdge{e.dst, NodePort(0), e.src, NodePort(0)};
  });
}

std::unordered_set<Node>
    ViewDiGraphAsMultiDiGraph::query_nodes(NodeQuery const &node_query) const {
  return this->g.query_nodes(node_query);
}

ViewMultiDiGraphAsOpenMultiDiGraph::ViewMultiDiGraphAsOpenMultiDiGraph(
    MultiDiGraphView const &g)
    : g(g) {}

std::unordered_set<OpenMultiDiEdge>
    ViewMultiDiGraphAsOpenMultiDiGraph::query_edges(
        OpenMultiDiEdgeQuery const &q) const {
  return transform(g.query_edges(q.standard_edge_query),
                   [](MultiDiEdge const &e) { return OpenMultiDiEdge(e); });
}

std::unordered_set<Node>
    ViewMultiDiGraphAsOpenMultiDiGraph::query_nodes(NodeQuery const &q) const {
  return g.query_nodes(q);
}

ViewMultiDiGraphAsOpenMultiDiGraph *
    ViewMultiDiGraphAsOpenMultiDiGraph::clone() const {
  return new ViewMultiDiGraphAsOpenMultiDiGraph(g);
}

std::unordered_set<InputMultiDiEdge>
    query_edge(std::unordered_set<InputMultiDiEdge> const &edges,
               InputMultiDiEdgeQuery const &q) {
  return filter(edges, [&](InputMultiDiEdge const &e) {
    return includes(q.dsts, e.dst) && includes(q.dstIdxs, e.dst_idx);
  });
}

std::unordered_set<OutputMultiDiEdge>
    query_edge(std::unordered_set<OutputMultiDiEdge> const &edges,
               OutputMultiDiEdgeQuery const &q) {
  return filter(edges, [&](OutputMultiDiEdge const &e) {
    return includes(q.srcs, e.src) && includes(q.srcIdxs, e.src_idx);
  });
}

OpenMultiDiSubgraphView::OpenMultiDiSubgraphView(
    OpenMultiDiGraphView const &g, std::unordered_set<Node> const &nodes)
    : g(g), nodes(nodes),
      inputs(transform(get_cut_set(g, nodes), to_inputmultidiedge)),
      outputs(transform(get_cut_set(g, nodes), to_outputmultidiedge)) {}

std::unordered_set<OpenMultiDiEdge>
    OpenMultiDiSubgraphView::query_edges(OpenMultiDiEdgeQuery const &q) const {
  OpenMultiDiEdgeQuery subgraph_query(
      q.input_edge_query.with_dst_nodes(nodes),
      q.standard_edge_query.with_src_nodes(nodes).with_dst_nodes(nodes),
      q.output_edge_query.with_src_nodes(nodes));
  std::unordered_set<OpenMultiDiEdge> result = g.query_edges(subgraph_query);
  extend(result, query_edge(inputs, q.input_edge_query.with_dst_nodes(nodes)));
  extend(result,
         query_edge(outputs, q.output_edge_query.with_src_nodes(nodes)));
  return result;
}

std::unordered_set<Node>
    OpenMultiDiSubgraphView::query_nodes(NodeQuery const &q) const {
  return g.query_nodes(query_intersection(q, NodeQuery(nodes)));
}

UpwardOpenMultiDiSubgraphView::UpwardOpenMultiDiSubgraphView(
    OpenMultiDiGraphView const &g, std::unordered_set<Node> const &nodes)
    : g(g), nodes(nodes),
      inputs(transform(get_cut_set(g, nodes), to_inputmultidiedge)) {}

UpwardOpenMultiDiSubgraphView *UpwardOpenMultiDiSubgraphView::clone() const {
  return new UpwardOpenMultiDiSubgraphView(g, nodes);
}

std::unordered_set<OpenMultiDiEdge> UpwardOpenMultiDiSubgraphView::query_edges(
    OpenMultiDiEdgeQuery const &q) const {
  OpenMultiDiEdgeQuery subgraph_query(
      q.input_edge_query.with_dst_nodes(nodes),
      q.standard_edge_query.with_src_nodes(nodes).with_dst_nodes(nodes),
      OutputMultiDiEdgeQuery::none());
  std::unordered_set<OpenMultiDiEdge> result = g.query_edges(subgraph_query);
  extend(result, query_edge(inputs, q.input_edge_query.with_dst_nodes(nodes)));
  return result;
}

std::unordered_set<Node>
    UpwardOpenMultiDiSubgraphView::query_nodes(NodeQuery const &q) const {
  return g.query_nodes(query_intersection(q, NodeQuery(nodes)));
}

DownwardOpenMultiDiSubgraphView::DownwardOpenMultiDiSubgraphView(
    OpenMultiDiGraphView const &g, std::unordered_set<Node> const &nodes)
    : g(g), nodes(nodes),
      outputs(transform(get_cut_set(g, nodes), to_outputmultidiedge)) {}

std::unordered_set<OpenMultiDiEdge>
    DownwardOpenMultiDiSubgraphView::query_edges(
        OpenMultiDiEdgeQuery const &q) const {
  OpenMultiDiEdgeQuery subgraph_query(
      InputMultiDiEdgeQuery::none(),
      q.standard_edge_query.with_src_nodes(nodes).with_dst_nodes(nodes),
      q.output_edge_query.with_src_nodes(nodes));
  std::unordered_set<OpenMultiDiEdge> result = g.query_edges(subgraph_query);
  extend(result,
         query_edge(outputs, q.output_edge_query.with_src_nodes(nodes)));
  return result;
}

std::unordered_set<Node>
    DownwardOpenMultiDiSubgraphView::query_nodes(NodeQuery const &q) const {
  return g.query_nodes(query_intersection(q, NodeQuery(nodes)));
}

ClosedMultiDiSubgraphView::ClosedMultiDiSubgraphView(
    OpenMultiDiGraphView const &g, std::unordered_set<Node> const &nodes)
    : g(g), nodes(nodes) {}

std::unordered_set<OpenMultiDiEdge> ClosedMultiDiSubgraphView::query_edges(
    OpenMultiDiEdgeQuery const &q) const {
  return g.query_edges(
      q.standard_edge_query.with_src_nodes(nodes).with_dst_nodes(nodes));
}

std::unordered_set<Node>
    ClosedMultiDiSubgraphView::query_nodes(NodeQuery const &q) const {
  return g.query_nodes(query_intersection(q, NodeQuery(nodes)));
}

ClosedMultiDiSubgraphView *ClosedMultiDiSubgraphView::clone() const {
  return new ClosedMultiDiSubgraphView(g, nodes);
}

JoinedUndirectedGraphView *JoinedUndirectedGraphView::clone() const {
  return new JoinedUndirectedGraphView(lhs, rhs);
}

DownwardOpenMultiDiSubgraphView *
    DownwardOpenMultiDiSubgraphView::clone() const {
  return new DownwardOpenMultiDiSubgraphView(g, nodes);
}

ViewDiGraphAsMultiDiGraph *ViewDiGraphAsMultiDiGraph::clone() const {
  return new ViewDiGraphAsMultiDiGraph(g);
}

OpenMultiDiSubgraphView *OpenMultiDiSubgraphView::clone() const {
  return new OpenMultiDiSubgraphView(g, nodes);
}

MultiDiSubgraphView *MultiDiSubgraphView::clone() const {
  return new MultiDiSubgraphView(g, subgraph_nodes);
}

} // namespace FlexFlow

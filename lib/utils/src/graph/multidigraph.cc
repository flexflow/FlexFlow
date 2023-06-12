#include "utils/graph/multidigraph.h"
#include "utils/containers.h"

namespace FlexFlow {

MultiDiEdge::MultiDiEdge(Node src, Node dst, NodePort srcIdx, NodePort dstIdx)
    : src(src), dst(dst), srcIdx(srcIdx), dstIdx(dstIdx) {}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_nodes(
    std::unordered_set<Node> const &nodes) const {
  MultiDiEdgeQuery e{*this};
  if (e.srcs != tl::nullopt) {
    throw std::runtime_error("expected srcs == tl::nullopt");
  }
  e.srcs = nodes;
  return e;
}

//add MultiDiEdgeQuery::MultiDiEdgeQuery constructor
MultiDiEdgeQuery::  MultiDiEdgeQuery(
      tl::optional<std::unordered_set<Node>> const &srcs,
      tl::optional<std::unordered_set<Node>> const &dsts,
      tl::optional<std::unordered_set<NodePort>> const &srcIdxs ,
      tl::optional<std::unordered_set<NodePort>> const &dstIdxs )
      :srcs(srcs), dsts(dsts),srcIdxs(srcIdxs), dstIdxs(dstIdxs)
{}


MultiDiEdgeQuery query_intersection(MultiDiEdgeQuery const &lhs, MultiDiEdgeQuery const &rhs){
  assert (lhs.srcs.has_value() && lhs.dsts.has_value() && rhs.srcs.has_value() && rhs.dsts.has_value());

  tl::optional<std::unordered_set<Node>> srcs = intersection(*lhs.srcs, *rhs.srcs);
  tl::optional<std::unordered_set<Node>> dsts = intersection(*lhs.dsts, *rhs.dsts);

  //TODO, how to set srcIdxs, dstIdxs
  return MultiDiEdgeQuery(srcs, dsts);
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_node(Node const &n) const {
  return this->with_src_nodes({n});
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_nodes(
    std::unordered_set<Node> const &nodes) const {
  MultiDiEdgeQuery e{*this};
  if (e.dsts != tl::nullopt) {
    throw std::runtime_error("expected dsts == tl::nullopt");
  }
  e.dsts = nodes;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_node(Node const &n) const {
  return this->with_dst_nodes({n});
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_idxs(
    std::unordered_set<NodePort> const &idxs) const {
  MultiDiEdgeQuery e{*this};
  if (e.srcIdxs != tl::nullopt) {
    throw std::runtime_error("expected srcIdxs == tl::nullopt");
  }
  e.srcIdxs = idxs;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_idx(NodePort const &idx) const {
  return this->with_src_idxs({idx});
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_idxs(
    std::unordered_set<NodePort> const &idxs) const {
  MultiDiEdgeQuery e{*this};
  if (e.dstIdxs != tl::nullopt) {
    throw std::runtime_error("expected dstIdxs == tl::nullopt");
  }
  e.dstIdxs = idxs;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_idx(NodePort const &idx) const {
  return this->with_dst_idxs({idx});
}

MultiDiEdgeQuery MultiDiEdgeQuery::all() {
  return MultiDiEdgeQuery{};
}

void swap(MultiDiGraphView &lhs, MultiDiGraphView &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

NodePort IMultiDiGraph::add_node_port(){
  NodePort np{this->next_node_idx};
  this->next_node_idx += 1;
  return np;
}

void IMultiDiGraph::add_node_port_unsafe(NodePort const &np) {
  this->next_node_idx = std::max(this->next_node_idx, np.value() + 1);
}

MultiDiGraph::MultiDiGraph(MultiDiGraph const &other) : ptr(other.ptr) {}

MultiDiGraph &MultiDiGraph::operator=(MultiDiGraph other) {
  swap(*this, other);
  return *this;
}

std::unordered_set<MultiDiEdge> MultiDiGraphView::query_edges(MultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

void swap(MultiDiGraph &lhs, MultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

MultiDiGraph::operator MultiDiGraphView() const {
  return MultiDiGraphView(this->ptr.get_mutable());
}

Node MultiDiGraph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

void MultiDiGraph::add_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->add_node_unsafe(n);
}

void MultiDiGraph::add_node_port_unsafe(NodePort const &np) {
  return this->ptr.get_mutable()->add_node_port_unsafe(np);
}

void MultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->remove_node_unsafe(n);
}

void MultiDiGraph::add_edge(MultiDiEdge const &e) {
  return this->ptr.get_mutable()->add_edge(e);
}

void MultiDiGraph::remove_edge(MultiDiEdge const &e) {
  return this->ptr.get_mutable()->remove_edge(e);
}

std::unordered_set<MultiDiEdge>
    MultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

}
#include "utils/graph/multidigraph.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

MultiDiEdge::MultiDiEdge(Node src, Node dst, size_t srcIdx, size_t dstIdx)
  : src(src), dst(dst), srcIdx(srcIdx), dstIdx(dstIdx)
{ }

bool MultiDiEdge::operator==(MultiDiEdge const &other) const {
  return visit_eq(*this, other);
}

bool MultiDiEdge::operator<(MultiDiEdge const &other) const {
  return visit_lt(*this, other);
}

std::ostream &operator<<(std::ostream &s, MultiDiEdge const &e) {
  return (s << "MultiDiEdge<" << e.src.idx << ":" << e.srcIdx << " -> " << e.dst.idx << ":" << e.dstIdx << ">");
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_nodes(std::unordered_set<Node> const &nodes) const {
  MultiDiEdgeQuery e{*this};
  if (e.srcs != tl::nullopt) {
    throw std::runtime_error("expected srcs == tl::nullopt");
  }
  e.srcs = nodes;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_node(Node const &n) const {
  return this->with_src_nodes({n});
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_nodes(std::unordered_set<Node> const &nodes) const {
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

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_idxs(std::unordered_set<std::size_t> const &idxs) const {
  MultiDiEdgeQuery e{*this};
  if (e.srcIdxs != tl::nullopt) {
    throw std::runtime_error("expected srcIdxs == tl::nullopt");
  }
  e.srcIdxs = idxs;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_idx(std::size_t idx) const {
  return this->with_src_idxs({idx});
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_idxs(std::unordered_set<std::size_t> const &idxs) const {
  MultiDiEdgeQuery e{*this};
  if (e.dstIdxs != tl::nullopt) {
    throw std::runtime_error("expected dstIdxs == tl::nullopt");
  }
  e.dstIdxs = idxs;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_idx(std::size_t idx) const {
  return this->with_dst_idxs({idx});
}

MultiDiEdgeQuery MultiDiEdgeQuery::all() {
  return MultiDiEdgeQuery{};
}

MultiDiGraph::MultiDiGraph(MultiDiGraph const &other) 
  : ptr(other.ptr->clone())
{ }

MultiDiGraph &MultiDiGraph::operator=(MultiDiGraph other) {
  swap(*this, other);
  return *this;
}

void swap(MultiDiGraph &lhs, MultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node MultiDiGraph::add_node() {
  return this->ptr->add_node();
}

void MultiDiGraph::add_node_unsafe(Node const &n) {
  return this->ptr->add_node_unsafe(n);
}

void MultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->ptr->remove_node_unsafe(n);
}

void MultiDiGraph::add_edge(MultiDiEdge const &e) {
  return this->ptr->add_edge(e);
}

void MultiDiGraph::remove_edge(MultiDiEdge const &e) {
  return this->ptr->remove_edge(e);
}

std::unordered_set<MultiDiEdge> MultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

MultiDiGraph::MultiDiGraph(std::unique_ptr<IMultiDiGraph> _ptr)
  : ptr(std::move(_ptr))
{ }

}

namespace std {
using ::FlexFlow::MultiDiEdge;

std::size_t hash<MultiDiEdge>::operator()(MultiDiEdge const &e) const {
  return visit_hash(e);
}
}

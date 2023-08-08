#include "utils/graph/multidigraph.h"

namespace FlexFlow {

MultiDiInput get_input(MultiDiEdge const &e) {
  return {e.dst, e.dstIdx};
}

MultiDiOutput get_output(MultiDiEdge const &e) {
  return {e.src, e.srcIdx};
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_src_nodes(query_set<Node> const &nodes) const {
  MultiDiEdgeQuery e = *this;
  if (!is_matchall(e.srcs)) {
    throw mk_runtime_error("Expected matchall previous value");
  }
  e.srcs = nodes;
  return e;
}

std::ostream &operator<<(std::ostream &os, MultiDiEdge const &edge) {
  return os << "MultiDiEdge{" << edge.src.value() << "," << edge.dst.value()
            << "," << edge.srcIdx.value() << "," << edge.dstIdx.value() << "}";
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_dst_nodes(query_set<Node> const &nodes) const {
  MultiDiEdgeQuery e = *this;
  if (!is_matchall(e.dsts)) {
    throw mk_runtime_error("Expected matchall previous value");
  }
  e.dsts = nodes;
  return e;
}

MultiDiEdgeQuery query_intersection(MultiDiEdgeQuery const &lhs,
                                    MultiDiEdgeQuery const &rhs) {
  std::unordered_set<Node> srcs_t1;
  if (is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    srcs_t1 = allowed_values(rhs.srcs);
  } else if (!is_matchall(lhs.srcs) && is_matchall(rhs.srcs)) {
    srcs_t1 = allowed_values(lhs.srcs);
  } else if (!is_matchall(lhs.srcs) && !is_matchall(rhs.srcs)) {
    srcs_t1 = allowed_values(query_intersection(lhs.srcs, rhs.srcs));
  }

  std::unordered_set<Node> dsts_t1;
  if (is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    dsts_t1 = allowed_values(rhs.dsts);
  } else if (!is_matchall(lhs.dsts) && is_matchall(rhs.dsts)) {
    dsts_t1 = allowed_values(lhs.dsts);
  } else if (!is_matchall(lhs.dsts) && !is_matchall(rhs.dsts)) {
    dsts_t1 = allowed_values(query_intersection(lhs.dsts, rhs.dsts));
  }

  std::unordered_set<NodePort> srcIdxs_t1;
  if (is_matchall(lhs.srcIdxs) && !is_matchall(rhs.srcIdxs)) {
    srcIdxs_t1 = allowed_values(rhs.srcIdxs);
  } else if (!is_matchall(lhs.srcIdxs) && is_matchall(rhs.srcIdxs)) {
    srcIdxs_t1 = allowed_values(lhs.srcIdxs);
  } else if (!is_matchall(lhs.srcIdxs) && !is_matchall(rhs.srcIdxs)) {
    srcIdxs_t1 = allowed_values(query_intersection(lhs.srcIdxs, rhs.srcIdxs));
  }

  std::unordered_set<NodePort> dstIdxs_t1;
  if (is_matchall(lhs.dstIdxs) && !is_matchall(rhs.dstIdxs)) {
    dstIdxs_t1 = allowed_values(rhs.dstIdxs);
  } else if (!is_matchall(lhs.dstIdxs) && is_matchall(rhs.dstIdxs)) {
    dstIdxs_t1 = allowed_values(lhs.dstIdxs);
  } else if (!is_matchall(lhs.dstIdxs) && !is_matchall(rhs.dstIdxs)) {
    dstIdxs_t1 = allowed_values(query_intersection(lhs.dstIdxs, rhs.dstIdxs));
  }

  MultiDiEdgeQuery e = MultiDiEdgeQuery::all();
  e.srcs = srcs_t1;
  e.dsts = dsts_t1;
  e.srcIdxs = srcIdxs_t1;
  e.dstIdxs = dstIdxs_t1;
  return e;
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_src_idxs(query_set<NodePort> const &idxs) const {
  MultiDiEdgeQuery e{*this};
  if (!is_matchall(e.srcIdxs)) {
    throw mk_runtime_error("Expected matchall previous value");
  }
  e.srcIdxs = idxs;
  return e;
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_dst_idxs(query_set<NodePort> const &idxs) const {
  MultiDiEdgeQuery e = *this;
  if (!is_matchall(e.dstIdxs)) {
    throw mk_runtime_error("Expected matchall previous value");
  }
  e.dstIdxs = idxs;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::all() {
  return {matchall<Node>(),
          matchall<Node>(),
          matchall<NodePort>(),
          matchall<NodePort>()};
}

std::unordered_set<Node>
    MultiDiGraphView::query_nodes(NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}

std::unordered_set<MultiDiEdge>
    MultiDiGraphView::query_edges(MultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

MultiDiGraphView::operator GraphView() const {
  return GraphView(this->ptr, should_only_be_used_internally_tag_t{});
}

// Set the shared_ptr's destructor to a nop so that effectively there is no
// ownership
MultiDiGraphView MultiDiGraphView::unsafe_create_without_ownership(
    IMultiDiGraphView const &graphView) {
  std::shared_ptr<IMultiDiGraphView const> ptr(
      (&graphView), [](IMultiDiGraphView const *ptr) {});
  return MultiDiGraphView(ptr);
}

void swap(MultiDiGraphView &lhs, MultiDiGraphView &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

void swap(MultiDiGraph &lhs, MultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

MultiDiGraph::operator MultiDiGraphView() const {
  return MultiDiGraphView::unsafe_create_without_ownership(*this->ptr);
}

Node MultiDiGraph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

std::vector<Node> MultiDiGraph::add_nodes(size_t n) {
  std::vector<Node> nodes;
  for (size_t i = 0; i < n; i++) {
    nodes.push_back(add_node());
  }
  return nodes;
}

std::vector<NodePort> MultiDiGraph::add_node_ports(size_t n) {
  std::vector<NodePort> ports;
  for (size_t i = 0; i < n; i++) {
    ports.push_back(add_node_port());
  }
  return ports;
}

NodePort MultiDiGraph::add_node_port() {
  return this->ptr.get_mutable()->add_node_port();
}

void MultiDiGraph::add_node_port_unsafe(NodePort const &np) {
  return this->ptr.get_mutable()->add_node_port_unsafe(np);
}

void MultiDiGraph::add_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->add_node_unsafe(n);
}

void MultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->remove_node_unsafe(n);
}

void MultiDiGraph::add_edge(MultiDiEdge const &e) {
  return this->ptr.get_mutable()->add_edge(e);
}

void MultiDiGraph::add_edges(std::vector<MultiDiEdge> const &edges) {
  for (MultiDiEdge const &e : edges) {
    add_edge(e);
  }
}

void MultiDiGraph::remove_edge(MultiDiEdge const &e) {
  return this->ptr.get_mutable()->remove_edge(e);
}

std::unordered_set<MultiDiEdge>
    MultiDiGraph::query_edges(MultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

std::unordered_set<Node> MultiDiGraph::query_nodes(NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}

} // namespace FlexFlow

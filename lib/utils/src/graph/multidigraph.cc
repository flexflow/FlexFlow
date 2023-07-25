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
  // if (is_matchall(e.srcs)) {
  //   throw mk_runtime_error("Expected matchall previous value");
  // }
  e.srcs = nodes;
  return e;
}

std::ostream &operator<<(std::ostream &os, MultiDiEdge const &edge) {
  return os << "MultiDiEdge{" << edge.src.value() << "," << edge.dst.value()
            << "," << edge.srcIdx.value() << "," << edge.dstIdx.value() << "}";
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_node(Node const &n) const {
  return this->with_src_nodes({n});
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_dst_nodes(query_set<Node> const &nodes) const {
  MultiDiEdgeQuery e = *this;
  // if (is_matchall(e.dsts)) {
  //   throw mk_runtime_error("Expected matchall previous value");
  // }
  e.dsts = nodes;
  return e;
}

MultiDiEdgeQuery query_intersection(MultiDiEdgeQuery const &lhs,
                                    MultiDiEdgeQuery const &rhs) {
  assert(lhs != tl::nullopt);
  assert(rhs != tl::nullopt);

  std::unordered_set<Node> srcs_t1 =
      intersection(allowed_values(lhs.srcs), allowed_values(rhs.srcs));
  std::unordered_set<Node> dsts_t1 =
      intersection(allowed_values(lhs.dsts), allowed_values(rhs.dsts));

  std::unordered_set<NodePort> srcIdxs_t1 =
      intersection(allowed_values(lhs.srcIdxs), allowed_values(rhs.srcIdxs));
  std::unordered_set<NodePort> dstIdxs_t1 =
      intersection(allowed_values(lhs.dstIdxs), allowed_values(rhs.dstIdxs));

  MultiDiEdgeQuery e = MultiDiEdgeQuery::all();
  e.srcs = srcs_t1;
  e.dsts = dsts_t1;
  e.srcIdxs = srcIdxs_t1;
  e.dstIdxs = dstIdxs_t1;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_node(Node const &n) const {
  return this->with_dst_nodes({n});
}
MultiDiEdgeQuery MultiDiEdgeQuery::with_src_idx(NodePort const &p) const {
  return this->with_src_idxs({p});
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_idx(NodePort const &p) const {
  return this->with_dst_idxs({p});
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_src_idxs(query_set<NodePort> const &idxs) const {
  MultiDiEdgeQuery e{*this};
  // if (is_matchall(e.srcIdxs)) {
  //   throw mk_runtime_error("Expected matchall previous value");
  // }
  e.srcIdxs = idxs;
  return e;
}

MultiDiEdgeQuery
    MultiDiEdgeQuery::with_dst_idxs(query_set<NodePort> const &idxs) const {
  MultiDiEdgeQuery e = *this;
  // if (is_matchall(e.dstIdxs)) {
  //   throw mk_runtime_error("Expected matchall previous value");
  // }
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

NodePort IMultiDiGraph::add_node_port() {
  NodePort np{this->next_nodeport_idx};
  this->next_nodeport_idx += 1;
  return np;
}

void IMultiDiGraph::add_node_port_unsafe(NodePort const &np) {
  this->next_nodeport_idx = std::max(this->next_nodeport_idx, np.value() + 1);
}

MultiDiGraphView::operator GraphView() const {
  return GraphView::unsafe_create(*(this->ptr.get()));
}

MultiDiGraphView
    MultiDiGraphView::unsafe_create(IMultiDiGraphView const &graphView) {
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
  return MultiDiGraphView::unsafe_create(*this->ptr);
}

Node MultiDiGraph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

std::vector<Node> MultiDiGraph::add_nodes(size_t n) {
  std::vector<Node> nodes;
  for(size_t i = 0; i < n; i++) {
    nodes.push_back(add_node());
  }
  return nodes;
}

std::vector<NodePort> MultiDiGraph::add_node_ports(size_t n) {
  std::vector<NodePort> ports;
  for(size_t i = 0; i < n; i++) {
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
  for(MultiDiEdge const &e : edges) {
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

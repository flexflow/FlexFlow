#include "utils/graph/open_graphs.h"
#include "utils/graph/query_set.h"

namespace FlexFlow {

InputMultiDiEdgeQuery InputMultiDiEdgeQuery::all(){
  return {matchall<Node>(), matchall<NodePort>()};
}

 OutputMultiDiEdgeQuery  OutputMultiDiEdgeQuery::all() {
  return {matchall<Node>(), matchall<NodePort>()};
 }

std::unordered_set<Node>
    OpenMultiDiGraphView::query_nodes(NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}
std::unordered_set<OpenMultiDiEdge>
    OpenMultiDiGraphView::query_edges(OpenMultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

OpenMultiDiGraph::OpenMultiDiGraph(OpenMultiDiGraph const &other)
    : ptr(other.ptr) {}

OpenMultiDiGraph &OpenMultiDiGraph::operator=(OpenMultiDiGraph other) {
  swap(*this, other);
  return *this;
}

void swap(OpenMultiDiGraph &lhs, OpenMultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node OpenMultiDiGraph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

void OpenMultiDiGraph::add_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->add_node_unsafe(n);
}

void OpenMultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->remove_node_unsafe(n);
}

void OpenMultiDiGraph::add_edge(OpenMultiDiEdge const &e) {
  return this->ptr.get_mutable()->add_edge(e);
}

void OpenMultiDiGraph::remove_edge(OpenMultiDiEdge const &e) {
  return this->ptr.get_mutable()->remove_edge(e);
}

std::unordered_set<OpenMultiDiEdge>
    OpenMultiDiGraph::query_edges(OpenMultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

OpenMultiDiGraph::OpenMultiDiGraph(std::unique_ptr<IOpenMultiDiGraph> _ptr)
    : ptr(std::move(_ptr)) {}

UpwardOpenMultiDiGraph &
    UpwardOpenMultiDiGraph::operator=(UpwardOpenMultiDiGraph other) {
  swap(*this, other);
  return *this;
}

void swap(UpwardOpenMultiDiGraph &lhs, UpwardOpenMultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node UpwardOpenMultiDiGraph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

void UpwardOpenMultiDiGraph::add_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->add_node_unsafe(n);
}

void UpwardOpenMultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->remove_node_unsafe(n);
}

void UpwardOpenMultiDiGraph::add_edge(UpwardOpenMultiDiEdge const &e) {
  return this->ptr.get_mutable()->add_edge(e);
}

void UpwardOpenMultiDiGraph::remove_edge(UpwardOpenMultiDiEdge const &e) {
  return this->ptr.get_mutable()->remove_edge(e);
}

std::unordered_set<UpwardOpenMultiDiEdge> UpwardOpenMultiDiGraph::query_edges(
    UpwardOpenMultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

UpwardOpenMultiDiGraph::UpwardOpenMultiDiGraph(
    std::unique_ptr<IUpwardOpenMultiDiGraph> _ptr)
    : ptr(std::move(_ptr)) {}

DownwardOpenMultiDiGraph &
    DownwardOpenMultiDiGraph::operator=(DownwardOpenMultiDiGraph other) {
  swap(*this, other);
  return *this;
}

void swap(DownwardOpenMultiDiGraph &lhs, DownwardOpenMultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node DownwardOpenMultiDiGraph::add_node() {
  return this->ptr.get_mutable()->add_node();
}

void DownwardOpenMultiDiGraph::add_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->add_node_unsafe(n);
}

void DownwardOpenMultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->ptr.get_mutable()->remove_node_unsafe(n);
}

void DownwardOpenMultiDiGraph::add_edge(DownwardOpenMultiDiEdge const &e) {
  return this->ptr.get_mutable()->add_edge(e);
}

void DownwardOpenMultiDiGraph::remove_edge(DownwardOpenMultiDiEdge const &e) {
  return this->ptr.get_mutable()->remove_edge(e);
}

std::unordered_set<DownwardOpenMultiDiEdge>
    DownwardOpenMultiDiGraph::query_edges(
        DownwardOpenMultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

DownwardOpenMultiDiGraph::DownwardOpenMultiDiGraph(
    std::unique_ptr<IDownwardOpenMultiDiGraph> _ptr)
    : ptr(std::move(_ptr)) {}

} // namespace FlexFlow

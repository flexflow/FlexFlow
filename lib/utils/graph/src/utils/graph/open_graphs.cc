#include "utils/graph/open_graphs.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/internal.h"
#include "utils/graph/multidigraph.h"
#include "utils/graph/query_set.h"

namespace FlexFlow {

void swap(OpenMultiDiGraphView &lhs, OpenMultiDiGraphView &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

std::unordered_set<Node>
    OpenMultiDiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}
std::unordered_set<OpenMultiDiEdge>
    OpenMultiDiGraphView::query_edges(OpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

OpenMultiDiGraphView::OpenMultiDiGraphView(
    cow_ptr_t<IOpenMultiDiGraphView> ptr)
    : MultiDiGraphView(ptr) {}

cow_ptr_t<IOpenMultiDiGraphView> OpenMultiDiGraphView::get_ptr() const {
  return static_cast<cow_ptr_t<IOpenMultiDiGraphView>>(ptr);
}

void swap(OpenMultiDiGraph &lhs, OpenMultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node OpenMultiDiGraph::add_node() {
  return this->get_ptr().get_mutable()->add_node();
}

void OpenMultiDiGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->add_node_unsafe(n);
}

void OpenMultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->remove_node_unsafe(n);
}

void OpenMultiDiGraph::add_edge(OpenMultiDiEdge const &e) {
  return this->get_ptr().get_mutable()->add_edge(e);
}

void OpenMultiDiGraph::remove_edge(OpenMultiDiEdge const &e) {
  return this->get_ptr().get_mutable()->remove_edge(e);
}

std::unordered_set<OpenMultiDiEdge>
    OpenMultiDiGraph::query_edges(OpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

OpenMultiDiGraph::OpenMultiDiGraph(cow_ptr_t<IOpenMultiDiGraph> _ptr)
    : OpenMultiDiGraphView(_ptr) {}

cow_ptr_t<IOpenMultiDiGraph> OpenMultiDiGraph::get_ptr() const {
  return static_cast<cow_ptr_t<IOpenMultiDiGraph>>(ptr);
}

void swap(UpwardOpenMultiDiGraphView &lhs, UpwardOpenMultiDiGraphView &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

std::unordered_set<Node> UpwardOpenMultiDiGraphView::query_nodes(NodeQuery const &q) {
  return get_ptr()->query_nodes(q);
}

std::unordered_set<UpwardOpenMultiDiEdge> UpwardOpenMultiDiGraphView::query_edges(UpwardOpenMultiDiEdgeQuery const &q) {
  return get_ptr()->query_edges(q);
}

UpwardOpenMultiDiGraphView::UpwardOpenMultiDiGraphView(
    cow_ptr_t<IUpwardOpenMultiDiGraphView> ptr) : MultiDiGraphView(ptr) {}

cow_ptr_t<IUpwardOpenMultiDiGraphView> UpwardOpenMultiDiGraphView::get_ptr() const {
  return static_cast<cow_ptr_t<IUpwardOpenMultiDiGraphView>>(ptr);
}

void swap(UpwardOpenMultiDiGraph &lhs, UpwardOpenMultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node UpwardOpenMultiDiGraph::add_node() {
  return this->get_ptr().get_mutable()->add_node();
}

void UpwardOpenMultiDiGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->add_node_unsafe(n);
}

void UpwardOpenMultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->remove_node_unsafe(n);
}

void UpwardOpenMultiDiGraph::add_edge(UpwardOpenMultiDiEdge const &e) {
  return this->get_ptr().get_mutable()->add_edge(e);
}

void UpwardOpenMultiDiGraph::remove_edge(UpwardOpenMultiDiEdge const &e) {
  return this->get_ptr().get_mutable()->remove_edge(e);
}

std::unordered_set<UpwardOpenMultiDiEdge> UpwardOpenMultiDiGraph::query_edges(
    UpwardOpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

UpwardOpenMultiDiGraph::UpwardOpenMultiDiGraph(
    cow_ptr_t<IUpwardOpenMultiDiGraph> _ptr)
    : UpwardOpenMultiDiGraphView(_ptr) {}

cow_ptr_t<IUpwardOpenMultiDiGraph> UpwardOpenMultiDiGraph::get_ptr() const {
  return static_cast<cow_ptr_t<IUpwardOpenMultiDiGraph>>(ptr);
}

void swap(DownwardOpenMultiDiGraphView &lhs, DownwardOpenMultiDiGraphView &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

std::unordered_set<Node> DownwardOpenMultiDiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

std::unordered_set<DownwardOpenMultiDiEdge>
    DownwardOpenMultiDiGraphView::query_edges(
        DownwardOpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

DownwardOpenMultiDiGraphView::DownwardOpenMultiDiGraphView(cow_ptr_t<IDownwardOpenMultiDiGraphView> ptr) : MultiDiGraphView(ptr) {}

cow_ptr_t<IDownwardOpenMultiDiGraphView> get_ptr() const {
  return static_cast<cow_ptr_t<IDownwardOpenMultiDiGraphView>>(ptr);
}

void swap(DownwardOpenMultiDiGraph &lhs, DownwardOpenMultiDiGraph &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

Node DownwardOpenMultiDiGraph::add_node() {
  return this->get_ptr().get_mutable()->add_node();
}

void DownwardOpenMultiDiGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->add_node_unsafe(n);
}

void DownwardOpenMultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().get_mutable()->remove_node_unsafe(n);
}

void DownwardOpenMultiDiGraph::add_edge(DownwardOpenMultiDiEdge const &e) {
  return this->get_ptr().get_mutable()->add_edge(e);
}

void DownwardOpenMultiDiGraph::remove_edge(DownwardOpenMultiDiEdge const &e) {
  return this->get_ptr().get_mutable()->remove_edge(e);
}

std::unordered_set<Node>
    DownwardOpenMultiDiGraph::query_nodes(
        NodeQuery const &q) const {
  return this->get_ptr()->query_nodes(q);
}

std::unordered_set<DownwardOpenMultiDiEdge>
    DownwardOpenMultiDiGraph::query_edges(
        DownwardOpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr()->query_edges(q);
}

DownwardOpenMultiDiGraph::DownwardOpenMultiDiGraph(
    cow_ptr_t<IDownwardOpenMultiDiGraph> _ptr)
    : DownwardOpenMultiDiGraphView(ptr) {}

cow_ptr_t<IDownwardOpenMultiDiGraph> DownwardOpenMultiDiGraph::get_ptr() const {
  return static_cast<cow_ptr_t<IDownwardOpenMultiDiGraph>>(ptr);
}

} // namespace FlexFlow

#include "utils/graph/open_graphs.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/multidigraph.h"
#include "utils/graph/query_set.h"

namespace FlexFlow {

std::unordered_set<MultiDiEdge>
    IOpenMultiDiGraphView::query_edges(MultiDiEdgeQuery const &q) const {
  return transform(
      query_edges(OpenMultiDiEdgeQuery(q)),
      [](OpenMultiDiEdge const &e) { return get<MultiDiEdge>(e); });
}

std::unordered_set<Node>
    OpenMultiDiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}
std::unordered_set<OpenMultiDiEdge>
    OpenMultiDiGraphView::query_edges(OpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

IOpenMultiDiGraphView const &OpenMultiDiGraphView::get_ptr() const {
  return *std::dynamic_pointer_cast<IOpenMultiDiGraphView const>(
      GraphView::ptr.get());
}

Node OpenMultiDiGraph::add_node() {
  return this->get_ptr().add_node();
}

void OpenMultiDiGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().add_node_unsafe(n);
}

void OpenMultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().remove_node_unsafe(n);
}

void OpenMultiDiGraph::add_edge(OpenMultiDiEdge const &e) {
  return this->get_ptr().add_edge(e);
}

void OpenMultiDiGraph::remove_edge(OpenMultiDiEdge const &e) {
  return this->get_ptr().remove_edge(e);
}

std::unordered_set<OpenMultiDiEdge>
    OpenMultiDiGraph::query_edges(OpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

NodePort OpenMultiDiGraph::add_node_port() {
  return get_ptr().add_node_port();
}

IOpenMultiDiGraph &OpenMultiDiGraph::get_ptr() {
  return *std::dynamic_pointer_cast<IOpenMultiDiGraph>(
      GraphView::ptr.get_mutable());
}

IOpenMultiDiGraph const &OpenMultiDiGraph::get_ptr() const {
  return *std::reinterpret_pointer_cast<IOpenMultiDiGraph const>(
      GraphView::ptr.get());
}

std::unordered_set<Node>
    UpwardOpenMultiDiGraphView::query_nodes(NodeQuery const &q) {
  return get_ptr().query_nodes(q);
}

std::unordered_set<UpwardOpenMultiDiEdge>
    UpwardOpenMultiDiGraphView::query_edges(
        UpwardOpenMultiDiEdgeQuery const &q) {
  return get_ptr().query_edges(q);
}

IUpwardOpenMultiDiGraphView const &UpwardOpenMultiDiGraphView::get_ptr() const {
  return *std::dynamic_pointer_cast<IUpwardOpenMultiDiGraphView const>(
      GraphView::ptr.get());
}

Node UpwardOpenMultiDiGraph::add_node() {
  return this->get_ptr().add_node();
}

void UpwardOpenMultiDiGraph::add_node_unsafe(Node const &n) {
  return this->get_ptr().add_node_unsafe(n);
}

void UpwardOpenMultiDiGraph::remove_node_unsafe(Node const &n) {
  return this->get_ptr().remove_node_unsafe(n);
}

void UpwardOpenMultiDiGraph::add_edge(UpwardOpenMultiDiEdge const &e) {
  return this->get_ptr().add_edge(e);
}

void UpwardOpenMultiDiGraph::remove_edge(UpwardOpenMultiDiEdge const &e) {
  return this->get_ptr().remove_edge(e);
}

std::unordered_set<UpwardOpenMultiDiEdge> UpwardOpenMultiDiGraph::query_edges(
    UpwardOpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

IUpwardOpenMultiDiGraph const &UpwardOpenMultiDiGraph::get_ptr() const {
  return *std::dynamic_pointer_cast<IUpwardOpenMultiDiGraph const>(
      GraphView::ptr.get());
}

IUpwardOpenMultiDiGraph &UpwardOpenMultiDiGraph::get_ptr() {
  return *std::dynamic_pointer_cast<IUpwardOpenMultiDiGraph>(
      GraphView::ptr.get_mutable());
}

std::unordered_set<Node>
    DownwardOpenMultiDiGraphView::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}

std::unordered_set<DownwardOpenMultiDiEdge>
    DownwardOpenMultiDiGraphView::query_edges(
        DownwardOpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

IDownwardOpenMultiDiGraphView const &
    DownwardOpenMultiDiGraphView::get_ptr() const {
  return *std::dynamic_pointer_cast<IDownwardOpenMultiDiGraphView const>(
      GraphView::ptr.get());
}

Node DownwardOpenMultiDiGraph::add_node() {
  return this->get_ptr().add_node();
}

void DownwardOpenMultiDiGraph::add_node_unsafe(Node const &n) {
  this->get_ptr().add_node_unsafe(n);
}

void DownwardOpenMultiDiGraph::remove_node_unsafe(Node const &n) {
  this->get_ptr().remove_node_unsafe(n);
}

void DownwardOpenMultiDiGraph::add_edge(DownwardOpenMultiDiEdge const &e) {
  this->get_ptr().add_edge(e);
}

void DownwardOpenMultiDiGraph::remove_edge(DownwardOpenMultiDiEdge const &e) {
  this->get_ptr().remove_edge(e);
}

std::unordered_set<Node>
    DownwardOpenMultiDiGraph::query_nodes(NodeQuery const &q) const {
  return this->get_ptr().query_nodes(q);
}

std::unordered_set<DownwardOpenMultiDiEdge>
    DownwardOpenMultiDiGraph::query_edges(
        DownwardOpenMultiDiEdgeQuery const &q) const {
  return this->get_ptr().query_edges(q);
}

IDownwardOpenMultiDiGraph &DownwardOpenMultiDiGraph::get_ptr() {
  return *std::dynamic_pointer_cast<IDownwardOpenMultiDiGraph>(
      GraphView::ptr.get_mutable());
}

IDownwardOpenMultiDiGraph const &DownwardOpenMultiDiGraph::get_ptr() const {
  return *std::reinterpret_pointer_cast<IDownwardOpenMultiDiGraph const>(
      GraphView::ptr.get());
}

} // namespace FlexFlow

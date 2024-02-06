#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_OPEN_H

#include "labelled_open.decl.h"
#include "labelled_open_interfaces.h"
#include "node_labelled.h"
#include "utils/graph/open_graph_interfaces.h"
#include "utils/graph/open_graphs.h"

namespace FlexFlow {

// LabelledOpenMultiDiGraphView
template <typename N, typename E, typename I, typename O>
LabelledOpenMultiDiGraphView<N, E, I, O>::operator OpenMultiDiGraphView()
    const {
  return GraphInternal::create_open_multidigraph_view(this->ptr);
}

// template <typename N, typename E, typename I, typename O>
// LabelledOpenMultiDiGraphView<N, E, I, O>::operator MultiDiGraphView() const {
//   return GraphInternal::create_multidigraphview(this->ptr);
// }

template <typename NodeLabel, typename E, typename I, typename O>
NodeLabel const &
    LabelledOpenMultiDiGraphView<NodeLabel, E, I, O>::at(Node const &n) const {
  return this->ptr->at(n);
}

template <typename N, typename EdgeLabel, typename I, typename O>
EdgeLabel const &LabelledOpenMultiDiGraphView<N, EdgeLabel, I, O>::at(
    MultiDiEdge const &e) const {
  return this->ptr->at(e);
}

template <typename N, typename E, typename InputLabel, typename O>
InputLabel const &LabelledOpenMultiDiGraphView<N, E, InputLabel, O>::at(
    InputMultiDiEdge const &e) const {
  return this->ptr->at(e);
}

template <typename N, typename E, typename I, typename OutputLabel>
OutputLabel const &LabelledOpenMultiDiGraphView<N, E, I, OutputLabel>::at(
    OutputMultiDiEdge const &e) const {
  return this->ptr->at(e);
}

template <typename N, typename E, typename I, typename O>
template <typename BaseImpl>
enable_if_t<std::is_base_of<
                typename LabelledOpenMultiDiGraphView<N, E, I, O>::Interface,
                BaseImpl>::value,
            LabelledOpenMultiDiGraphView<N, E, I, O>>
    LabelledOpenMultiDiGraphView<N, E, I, O>::create() {
  return LabelledOpenMultiDiGraphView<N, E, I, O>(std::make_shared<BaseImpl>());
}

// LabelledOpenMultiDiGraph
template <typename N, typename E, typename I, typename O>
LabelledOpenMultiDiGraph<N, E, I, O>::
    operator LabelledOpenMultiDiGraphView<N, E, I, O>() const {
  return GraphInternal::create_labelled_open_multidigraph_view<N, E, I, O>(
      this->ptr);
}

template <typename N, typename E, typename I, typename O>
LabelledOpenMultiDiGraph<N, E, I, O>::operator OpenMultiDiGraphView() const {
  return GraphInternal::create_open_multidigraph_view(this->ptr.get());
}

template <typename NodeLabel, typename E, typename I, typename O>
Node LabelledOpenMultiDiGraph<NodeLabel, E, I, O>::add_node(
    NodeLabel const &l) {
  return this->ptr.get_mutable()->add_node(l);
}

template <typename NodeLabel, typename E, typename I, typename O>
NodeLabel &LabelledOpenMultiDiGraph<NodeLabel, E, I, O>::at(Node const &n) {
  return this->ptr->at(n);
}

template <typename NodeLabel, typename E, typename I, typename O>
NodeLabel const &
    LabelledOpenMultiDiGraph<NodeLabel, E, I, O>::at(Node const &n) const {
  return this->ptr->ILabelledMultiDiGraph<NodeLabel, E>::at(n);
}

template <typename NodeLabel, typename E, typename I, typename O>
void LabelledOpenMultiDiGraph<NodeLabel, E, I, O>::add_node_unsafe(
    Node const &n, NodeLabel const &l) {
  this->ptr->add_node_unsafe(n, l);
}

template <typename N, typename E, typename I, typename O>
std::unordered_set<Node> LabelledOpenMultiDiGraph<N, E, I, O>::query_nodes(
    NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}

template <typename N, typename E, typename I, typename O>
std::unordered_set<OpenMultiDiEdge>
    LabelledOpenMultiDiGraph<N, E, I, O>::query_edges(
        OpenMultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

template <typename N, typename EdgeLabel, typename I, typename O>
void LabelledOpenMultiDiGraph<N, EdgeLabel, I, O>::add_edge(
    MultiDiEdge const &e, EdgeLabel const &l) {
  return this->ptr->add_edge(e, l);
}

template <typename N, typename EdgeLabel, typename I, typename O>
EdgeLabel &
    LabelledOpenMultiDiGraph<N, EdgeLabel, I, O>::at(MultiDiEdge const &e) {
  return this->ptr->at(e);
}

template <typename N, typename EdgeLabel, typename I, typename O>
EdgeLabel const &LabelledOpenMultiDiGraph<N, EdgeLabel, I, O>::at(
    MultiDiEdge const &e) const {
  return this->ptr->ILabelledMultiDiGraph<N, EdgeLabel>::at(e);
}

template <typename N, typename E, typename InputLabel, typename O>
void LabelledOpenMultiDiGraph<N, E, InputLabel, O>::add_edge(
    InputMultiDiEdge const &e, InputLabel const &l) {
  return this->ptr->add_edge(e, l);
}

template <typename N, typename E, typename InputLabel, typename O>
InputLabel &LabelledOpenMultiDiGraph<N, E, InputLabel, O>::at(
    InputMultiDiEdge const &e) {
  return this->ptr->at(e);
}

template <typename N, typename E, typename InputLabel, typename O>
InputLabel const &LabelledOpenMultiDiGraph<N, E, InputLabel, O>::at(
    InputMultiDiEdge const &e) const {
  return this->ptr->at(e);
}

template <typename N, typename E, typename I, typename OutputLabel>
void LabelledOpenMultiDiGraph<N, E, I, OutputLabel>::add_edge(
    OutputMultiDiEdge const &e, OutputLabel const &l) {
  return this->ptr->add_edge(e, l);
}

template <typename N, typename E, typename I, typename OutputLabel>
OutputLabel &LabelledOpenMultiDiGraph<N, E, I, OutputLabel>::at(
    OutputMultiDiEdge const &e) {
  return this->ptr->at(e);
}

template <typename N, typename E, typename I, typename OutputLabel>
OutputLabel const &LabelledOpenMultiDiGraph<N, E, I, OutputLabel>::at(
    OutputMultiDiEdge const &e) const {
  return this->ptr->at(e);
}

template <typename N, typename E, typename I, typename O>
template <typename BaseImpl>
enable_if_t<
    std::is_base_of<typename LabelledOpenMultiDiGraph<N, E, I, O>::Interface,
                    BaseImpl>::value,
    LabelledOpenMultiDiGraph<N, E, I, O>>
    LabelledOpenMultiDiGraph<N, E, I, O>::create() {
  return LabelledOpenMultiDiGraph<N, E, I, O>(make_cow_ptr<BaseImpl>());
}

} // namespace FlexFlow

#endif

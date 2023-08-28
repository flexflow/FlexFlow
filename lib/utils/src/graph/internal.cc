#include "utils/graph/internal.h"
#include "utils/graph/node.h"
#include "utils/graph/undirected.h"
#include <memory>

namespace FlexFlow {

MultiDiGraph
    GraphInternal::create_multidigraph(cow_ptr_t<IMultiDiGraph> ptr) {
  return {std::move(ptr)};
}

MultiDiGraphView GraphInternal::create_multidigraphview(
    std::shared_ptr<IMultiDiGraphView const> ptr) {
  return {std::move(ptr)};
}

OpenMultiDiGraphView GraphInternal::create_open_multidigraph_view(std::shared_ptr<IOpenMultiDiGraphView const> ptr) {
  return {std::move(ptr)};
}

OpenMultiDiGraph GraphInternal::create_open_multidigraph(cow_ptr_t<IOpenMultiDiGraph> ptr) {
  return {std::move(ptr)};
}

DiGraph GraphInternal::create_digraph(cow_ptr_t<IDiGraph> ptr) {
  return {std::move(ptr)};
}

DiGraphView
    GraphInternal::create_digraphview(std::shared_ptr<IDiGraphView const> ptr) {
  return {std::move(ptr)};
}

UndirectedGraph GraphInternal::create_undirectedgraph(
    cow_ptr_t<IUndirectedGraph> ptr) {
  return {std::move(ptr)};
}

UndirectedGraphView GraphInternal::create_undirectedgraphview(
    std::shared_ptr<IUndirectedGraphView const> ptr) {
  return {std::move(ptr)};
}

Graph GraphInternal::create_graph(cow_ptr_t<IGraph> ptr) {
  return {std::move(ptr)};
}

GraphView
    GraphInternal::create_graphview(std::shared_ptr<IGraphView const> ptr) {
  return {std::move(ptr)};
}

} // namespace FlexFlow

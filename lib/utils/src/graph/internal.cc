#include "utils/graph/internal.h"
#include "utils/graph/node.h"
#include "utils/graph/undirected.h"
#include <memory>

namespace FlexFlow {

MultiDiGraph
    GraphInternal::create_multidigraph(std::shared_ptr<IMultiDiGraph> ptr) {
  return {std::move(ptr)};
}

MultiDiGraphView GraphInternal::create_multidigraphview(
    std::shared_ptr<IMultiDiGraphView const> ptr) {
  return {std::move(ptr)};
}

DiGraph GraphInternal::create_digraph(std::shared_ptr<IDiGraph> ptr) {
  return {std::move(ptr)};
}

DiGraphView
    GraphInternal::create_digraphview(std::shared_ptr<IDiGraphView const> ptr) {
  return {std::move(ptr)};
}

UndirectedGraph GraphInternal::create_undirectedgraph(
    std::shared_ptr<IUndirectedGraph> ptr) {
  return {std::move(ptr)};
}

UndirectedGraphView GraphInternal::create_undirectedgraphview(
    std::shared_ptr<IUndirectedGraphView const> ptr) {
  return {std::move(ptr)};
}

Graph GraphInternal::create_graph(std::shared_ptr<IGraph> ptr) {
  return {std::move(ptr)};
}

GraphView
    GraphInternal::create_graphview(std::shared_ptr<IGraphView const> ptr) {
  return {std::move(ptr)};
}

} // namespace FlexFlow

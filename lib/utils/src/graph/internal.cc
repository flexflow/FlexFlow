#include "internal.h"
#include "utils/graph/node.h"
#include "utils/graph/undirected.h"
#include <memory>

namespace FlexFlow {

MultiDiGraph GraphInternal::create_multidigraph(std::unique_ptr<IMultiDiGraph> ptr) {
  return {std::move(ptr)};
}

MultiDiGraphView GraphInternal::create_multidigraphview(std::shared_ptr<IMultiDiGraphView const> ptr) {
  return {std::move(ptr)};
}

DiGraph GraphInternal::create_digraph(std::unique_ptr<IDiGraph> ptr) {
  return {std::move(ptr)};
}

DiGraphView GraphInternal::create_digraphview(std::shared_ptr<IDiGraphView const> ptr) {
  return {std::move(ptr)};
}

UndirectedGraph GraphInternal::create_undirectedgraph(std::unique_ptr<IUndirectedGraph> ptr) {
  return {std::move(ptr)};
}

UndirectedGraphView GraphInternal::create_undirectedgraphview(std::shared_ptr<IUndirectedGraphView const> ptr) {
  return {std::move(ptr)};
}

Graph create_graph(std::unique_ptr<IGraph> ptr) {
  return {std::move(ptr)};
}

GraphView GraphInternal::create_graphview(std::shared_ptr<IGraphView const> ptr) {
  return {std::move(ptr)};
}

}

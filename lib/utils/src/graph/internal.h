#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_INTERNAL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_INTERNAL_H

#include "utils/graph/multidigraph_interfaces.h"
#include "utils/graph/digraph_interfaces.h"
#include "utils/graph/undirected.h"
#include "utils/graph/multidigraph.h"
#include "utils/graph/digraph.h" 
#include "utils/graph/undirected.h"
#include "utils/graph/node.h"

namespace FlexFlow {

struct GraphInternal {
  static MultiDiGraph create_multidigraph(std::unique_ptr<IMultiDiGraph>);
  static MultiDiGraphView create_multidigraphview(std::shared_ptr<IMultiDiGraphView const>);

  static DiGraph create_digraph(std::unique_ptr<IDiGraph>);
  static DiGraphView create_digraphview(std::shared_ptr<IDiGraphView const>);

  static UndirectedGraph create_undirectedgraph(std::unique_ptr<IUndirectedGraph>);
  static UndirectedGraphView create_undirectedgraphview(std::shared_ptr<IUndirectedGraphView const>);

  static Graph create_graph(std::unique_ptr<IGraph>);
  static GraphView create_graphview(std::shared_ptr<IGraphView const>);
};

}

#endif

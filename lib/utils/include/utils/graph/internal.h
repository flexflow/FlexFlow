#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_INTERNAL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_INTERNAL_H

#include "utils/graph/digraph.h"
#include "utils/graph/digraph_interfaces.h"
#include "utils/graph/multidigraph.h"
#include "utils/graph/multidigraph_interfaces.h"
#include "utils/graph/node.h"
#include "utils/graph/open_graph_interfaces.h"
#include "utils/graph/open_graphs.h"
#include "utils/graph/undirected.h"
#include "utils/graph/labelled/labelled_open_interfaces.h"
#include "utils/graph/labelled/labelled_open.decl"

namespace FlexFlow {

struct GraphInternal {
private:
  static OpenMultiDiGraph create_open_multidigraph(std::shared_ptr<IOpenMultiDiGraph>);
  static OpenMultiDiGraphView create_open_multidigraph_view(std::shared_ptr<IOpenMultiDiGraphView const>);

  static MultiDiGraph create_multidigraph(std::shared_ptr<IMultiDiGraph>);
  static MultiDiGraphView
      create_multidigraphview(std::shared_ptr<IMultiDiGraphView const>);

  static DiGraph create_digraph(std::shared_ptr<IDiGraph>);
  static DiGraphView create_digraphview(std::shared_ptr<IDiGraphView const>);

  static UndirectedGraph
      create_undirectedgraph(std::shared_ptr<IUndirectedGraph>);
  static UndirectedGraphView
      create_undirectedgraphview(std::shared_ptr<IUndirectedGraphView const>);

  static Graph create_graph(std::shared_ptr<IGraph>);
  static GraphView create_graphview(std::shared_ptr<IGraphView const>);

  template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
  static LabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel, OutputLabel> create_labelled_open_multidigraph_view(std::shared_ptr<ILabelledOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel, OutputLabel> const>);

  friend struct MultiDiGraph;
  friend struct MultiDiGraphView;
  friend struct DiGraph;
  friend struct DiGraphView;
  friend struct UndirectedGraph;
  friend struct UndirectedGraphView;
  friend struct Graph;
  friend struct GraphView;
  friend struct OpenMultiDiGraphView;
  friend struct OpenMultiDiGraph;

  template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
  friend struct LabelledOpenMultiDiGraphView;

  template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel,
          typename OutputLabel>
  friend struct LabelledOpenMultiDiGraph;
};

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_DATAFLOW_GRAPH_DOWNWARD_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DOWNWARD_OPEN_DATAFLOW_GRAPH_DOWNWARD_OPEN_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/downward_open_dataflow_graph/dataflow_graph_output.dtg.h"
#include "utils/graph/downward_open_dataflow_graph/i_downward_open_dataflow_graph.h"

namespace FlexFlow {

struct DownwardOpenDataflowGraph : virtual DataflowGraph {
public:
  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<DataflowEdge> query_edges(DataflowEdgeQuery const &) const;
  std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &) const;
  std::vector<DataflowGraphOutput> get_graph_outputs() const;

protected:
  using DataflowGraph::DataflowGraph;

private:
  IDownwardOpenDataflowGraph &get_interface();
  IDownwardOpenDataflowGraph const &get_interface() const;
};

} // namespace FlexFlow

#endif

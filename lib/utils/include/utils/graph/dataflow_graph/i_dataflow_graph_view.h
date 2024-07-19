#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_I_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_I_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/dataflow_graph/dataflow_edge.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_edge_query.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_output.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_output_query.dtg.h"
#include "utils/graph/digraph/i_digraph_view.h"

namespace FlexFlow {

struct IDataflowGraphView : virtual public IDiGraphView {
  virtual std::unordered_set<DataflowEdge>
      query_edges(DataflowEdgeQuery const &) const = 0;
  virtual std::unordered_set<DataflowOutput>
      query_outputs(DataflowOutputQuery const &) const = 0;

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override final;

  virtual ~IDataflowGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IDataflowGraphView);

} // namespace FlexFlow

#endif

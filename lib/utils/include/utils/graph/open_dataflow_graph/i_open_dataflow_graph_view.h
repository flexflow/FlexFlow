#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_I_OPEN_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_I_OPEN_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/dataflow_graph/i_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/dataflow_graph_input.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.dtg.h"

namespace FlexFlow {

struct IOpenDataflowGraphView : virtual public IDataflowGraphView {
  virtual std::unordered_set<DataflowGraphInput> get_inputs() const = 0;
  virtual std::unordered_set<OpenDataflowEdge>
      query_edges(OpenDataflowEdgeQuery const &) const = 0;

  std::unordered_set<DataflowEdge>
      query_edges(DataflowEdgeQuery const &) const override final;

  virtual ~IOpenDataflowGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenDataflowGraphView);

} // namespace FlexFlow

#endif

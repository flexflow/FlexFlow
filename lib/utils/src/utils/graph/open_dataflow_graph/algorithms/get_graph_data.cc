#include "utils/graph/open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_edges.h"

namespace FlexFlow {

OpenDataflowGraphData get_graph_data(OpenDataflowGraphView const &g) {
  return OpenDataflowGraphData{
      get_nodes(g),
      get_edges(g),
      g.get_inputs(),
      get_all_dataflow_outputs(g),
  };
}

} // namespace FlexFlow

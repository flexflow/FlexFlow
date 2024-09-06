#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_graph_inputs.h"

namespace FlexFlow {

std::unordered_set<DataflowGraphInput>
    get_open_dataflow_graph_inputs(OpenDataflowGraphView const &g) {
  return g.get_inputs();
}

} // namespace FlexFlow

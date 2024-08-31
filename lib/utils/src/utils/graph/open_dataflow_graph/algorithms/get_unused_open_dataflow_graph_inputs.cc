#include "utils/graph/open_dataflow_graph/algorithms/get_unused_open_dataflow_graph_inputs.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_graph_inputs.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_value_uses.h"

namespace FlexFlow {

std::unordered_set<DataflowGraphInput> get_unused_open_dataflow_graph_inputs(OpenDataflowGraphView const &g) {
  return filter(get_open_dataflow_graph_inputs(g),
                [&](DataflowGraphInput const &i) {
                  return get_open_dataflow_value_uses(g, OpenDataflowValue{i}).empty();
                });
}

} // namespace FlexFlow

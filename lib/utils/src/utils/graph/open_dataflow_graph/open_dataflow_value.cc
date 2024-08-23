#include "utils/graph/open_dataflow_graph/open_dataflow_value.h"
#include "utils/overload.h"

namespace FlexFlow {

std::optional<DataflowOutput> try_get_dataflow_output(OpenDataflowValue const &v) {
  return v.visit<std::optional<DataflowOutput>>(overload {
    [](DataflowOutput const &o) { return o; },
    [](DataflowGraphInput const &i) { return std::nullopt; },
  });
}

std::optional<DataflowGraphInput> try_get_dataflow_graph_input(OpenDataflowValue const &v) {
  return v.visit<std::optional<DataflowGraphInput>>(overload {
    [](DataflowOutput const &o) { return std::nullopt; },
    [](DataflowGraphInput const &i) { return i; },
  });
}

} // namespace FlexFlow

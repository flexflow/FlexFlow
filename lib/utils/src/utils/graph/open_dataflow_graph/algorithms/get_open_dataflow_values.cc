#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowValue>
    get_open_dataflow_values(OpenDataflowGraphView const &g) {
  return set_union(
      transform(
          unordered_set_of(g.get_inputs()),
          [](DataflowGraphInput const &gi) { return OpenDataflowValue{gi}; }),
      transform(get_all_dataflow_outputs(g),
                [](DataflowOutput const &o) { return OpenDataflowValue{o}; }));
}

} // namespace FlexFlow

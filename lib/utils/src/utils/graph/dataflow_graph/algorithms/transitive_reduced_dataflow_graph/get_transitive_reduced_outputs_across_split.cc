#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/get_transitive_reduced_outputs_across_split.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/get_transitive_reduced_edges_across_split.h"

namespace FlexFlow {

std::unordered_set<DataflowOutput> get_transitive_reduced_outputs_across_split(
    TransitiveReducedDataflowGraphView const &tr_g,
    BinarySeriesSplit const &split) {
  return transform(get_transitive_reduced_edges_across_split(tr_g, split),
                   [](DataflowEdge const &e) { return e.src; });
}

} // namespace FlexFlow

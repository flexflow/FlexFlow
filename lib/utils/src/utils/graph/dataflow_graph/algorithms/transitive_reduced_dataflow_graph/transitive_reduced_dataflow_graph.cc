#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/transitive_reduced_dataflow_graph.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"

namespace FlexFlow {

TransitiveReducedDataflowGraphView
    get_dataflow_graph_transitive_reduction(DataflowGraphView const &g) {
  DiGraphView as_digraph = g;
  DiGraphView transitive_reduced = transitive_reduction(as_digraph);

  return TransitiveReducedDataflowGraphView{
      /*full_dataflow_graph=*/g,
      /*transitive_reduction=*/transitive_reduced,
  };
}

} // namespace FlexFlow

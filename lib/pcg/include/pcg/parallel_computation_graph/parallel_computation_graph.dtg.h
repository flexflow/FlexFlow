// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/parallel_computation_graph/parallel_computation_graph.struct.toml
/* proj-data
{
  "generated_from": "1339be6e86e9818c36d6ecf5475e2d4b"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_COMPUTATION_GRAPH_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_COMPUTATION_GRAPH_DTG_H

#include "pcg/dataflow_graph/dataflow_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"

namespace FlexFlow {
struct ParallelComputationGraph {
  ParallelComputationGraph() = delete;
  ParallelComputationGraph(
      ::FlexFlow::DataflowGraph<::FlexFlow::ParallelLayerAttrs,
                                ::FlexFlow::ParallelTensorAttrs> const
          &raw_graph);

  ::FlexFlow::DataflowGraph<::FlexFlow::ParallelLayerAttrs,
                            ::FlexFlow::ParallelTensorAttrs>
      raw_graph;
};
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_COMPUTATION_GRAPH_DTG_H

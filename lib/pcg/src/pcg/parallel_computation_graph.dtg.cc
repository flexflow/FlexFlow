// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/parallel_computation_graph.struct.toml
/* proj-data
{
  "generated_from": "e4db0f603f7b8947dda13e01f96c40fb"
}
*/

#include "pcg/parallel_computation_graph.dtg.h"

#include "pcg/dataflow_graph.h"
#include "pcg/parallel_layer_attrs.dtg.h"
#include "pcg/parallel_tensor_attrs.dtg.h"

namespace FlexFlow {
ParallelComputationGraph::ParallelComputationGraph(
    ::FlexFlow::DataflowGraph<::FlexFlow::ParallelLayerAttrs,
                              ::FlexFlow::ParallelTensorAttrs> const &raw_graph)
    : raw_graph(raw_graph) {}
} // namespace FlexFlow

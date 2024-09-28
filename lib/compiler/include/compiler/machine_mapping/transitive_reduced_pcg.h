#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_TRANSITIVE_REDUCED_PCG_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_TRANSITIVE_REDUCED_PCG_H

#include "compiler/machine_mapping/transitive_reduced_pcg.dtg.h"
#include "compiler/machine_mapping/pcg_split_boundary_layers.dtg.h"
#include "compiler/series_parallel/pcg_binary_series_split.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/transitive_reduced_dataflow_graph.dtg.h"

namespace FlexFlow {

TransitiveReducedDataflowGraphView get_underlying_transitive_reduced_dataflow_graph(TransitiveReducedPCG const &);

TransitiveReducedPCG pcg_get_transitive_reduction(ParallelComputationGraph const &);

std::unordered_set<ParallelComputationGraphEdge> 
  pcg_get_transitive_reduced_edges_across_split(TransitiveReducedPCG const &,
                                            PCGBinarySeriesSplit const &);

std::unordered_set<parallel_tensor_guid_t> 
  pcg_get_transitive_reduced_tensors_across_split(TransitiveReducedPCG const &,
                                              PCGBinarySeriesSplit const &);

PCGSplitBoundaryLayers pcg_get_transitive_reduced_boundary_layers_for_split(TransitiveReducedPCG const &,
                                               PCGBinarySeriesSplit const &);


} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_TRANSITIVE_REDUCED_PCG_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_TRANSITIVE_REDUCED_PCG_H

#include "compiler/machine_mapping/transitive_reduced_pcg.dtg.h"
#include "compiler/series_parallel/pcg_binary_series_split.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

TransitiveReducedPCG get_pcg_transitive_reduction(ParallelComputationGraph const &);

std::unordered_set<parallel_layer_guid_t> get_transitive_reduced_predecessors(TransitiveReducedPCG const &,
                                                                                parallel_layer_guid_t const &);
std::unordered_set<parallel_layer_guid_t> get_transitive_reduced_successors(TransitiveReducedPCG const &,
                                                                              parallel_layer_guid_t const &);

std::unordered_set<ParallelComputationGraphEdge> 
  get_transitive_reduced_edges_across_split(TransitiveReducedPCG const &,
                                            PCGBinarySeriesSplit const &);

std::unordered_set<parallel_tensor_guid_t> 
  get_transitive_reduced_tensors_across_split(TransitiveReducedPCG const &,
                                              PCGBinarySeriesSplit const &);

std::pair<
  std::unordered_set<parallel_layer_guid_t>,
  std::unordered_set<parallel_layer_guid_t>
> get_split_transitive_reduced_boundary_layers(TransitiveReducedPCG const &,
                                               PCGBinarySeriesSplit const &);


} // namespace FlexFlow

#endif

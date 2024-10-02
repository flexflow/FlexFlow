#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/values.h"

namespace FlexFlow {

AbstractedTensorSetMovement get_abstracted_tensor_set_movement_across_split(TransitiveReducedPCG const &tr_pcg,
                                                                 PCGBinarySeriesSplit const &split) {

  auto get_path_to_layer = [&](parallel_layer_guid_t const &l) {
    return get_only(find_paths_to_leaf(wrap_series_split(split), l));
  };
  
  std::unordered_set<ParallelComputationGraphEdge> 
    edges_across_split = pcg_get_transitive_reduced_edges_across_split(tr_pcg, split);

  auto get_movement_for_tensor = [&](parallel_tensor_guid_t const &t) {
    std::unordered_set<ParallelComputationGraphEdge> tensor_edges = filter(edges_across_split,
                  [&](ParallelComputationGraphEdge const &e) { return get_parallel_tensor(e) == t; });

    std::unordered_set<parallel_layer_guid_t> src_layers = 
      transform(tensor_edges,
                [&](ParallelComputationGraphEdge const &e) {
                  return get_src_layer(e);
                });

    std::unordered_set<parallel_layer_guid_t> dst_layers = 
      transform(tensor_edges,
                [&](ParallelComputationGraphEdge const &e) {
                  return get_dst_layer(e);
                });

    return AbstractedSingleTensorMovement{
      /*parallel_tensor_shape=*/get_parallel_tensor_shape(tr_pcg.full_pcg, t),
      /*src_machine_views=*/transform(src_layers, get_path_to_layer),
      /*dst_machine_views=*/transform(dst_layers, get_path_to_layer),
    };
  };

  std::unordered_map<parallel_tensor_guid_t, AbstractedSingleTensorMovement> single_tensor_movements =
    generate_map(pcg_get_transitive_reduced_tensors_across_split(tr_pcg, split),
                 get_movement_for_tensor);

  return AbstractedTensorSetMovement{
    values(single_tensor_movements),
  };
}

} // namespace FlexFlow

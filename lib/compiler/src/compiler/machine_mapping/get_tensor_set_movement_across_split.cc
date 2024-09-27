#include "compiler/machine_mapping/get_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/partial_machine_mapping.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/sum.h"
#include "utils/containers/values.h"

namespace FlexFlow {

TensorSetMovement get_tensor_set_movement_across_split(TransitiveReducedPCG const &tr_pcg,
                                                       PCGBinarySeriesSplit const &split,
                                                       PartialMachineMapping const &pre_mapping,
                                                       PartialMachineMapping const &post_mapping) {
  std::unordered_set<ParallelComputationGraphEdge> 
    edges_across_split = get_transitive_reduced_edges_across_split(tr_pcg, split);
  
  auto get_movement_for_tensor = [&](parallel_tensor_guid_t const &t) {
    std::unordered_set<ParallelComputationGraphEdge> tensor_edges = filter(edges_across_split,
                  [&](ParallelComputationGraphEdge const &e) { return get_parallel_tensor(e) == t; });

    std::unordered_set<MachineView> src_machine_views = 
      transform(tensor_edges,
                [&](ParallelComputationGraphEdge const &e) {
                  return get_machine_view_for_layer(pre_mapping, get_src_layer(e)).value();
                });

    std::unordered_set<MachineView> dst_machine_views = 
      transform(tensor_edges,
                [&](ParallelComputationGraphEdge const &e) {
                  return get_machine_view_for_layer(post_mapping, get_dst_layer(e)).value();
                });

    return SingleTensorMovement{
      /*parallel_tensor_shape=*/get_parallel_tensor_shape(tr_pcg.full_pcg, t),
      /*src_machine_views=*/src_machine_views,
      /*dst_machine_views=*/dst_machine_views,
    };
  };

  std::unordered_map<parallel_tensor_guid_t, SingleTensorMovement> single_tensor_movements =
    generate_map(get_transitive_reduced_tensors_across_split(tr_pcg, split),
                 get_movement_for_tensor);

  return TensorSetMovement{
    values(single_tensor_movements),
  };
}


} // namespace FlexFlow



#include "compiler/machine_mapping/estimate_cost_across_split.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/containers/keys.h"
#include "utils/containers/sum.h"

namespace FlexFlow {

float estimate_cost_across_split(TransitiveReducedPCG const &tr_pcg,
                                 CostEstimator const &cost_estimator,
                                 std::unordered_map<parallel_layer_guid_t, MachineView> const &pre_machine_views,
                                 std::unordered_map<parallel_layer_guid_t, MachineView> const &post_machine_views) {
  std::unordered_set<ParallelComputationGraphEdge> 
    edges_across_split = get_transitive_reduced_edges_across_split(tr_pcg,
                                                                   keys(pre_machine_views),
                                                                   keys(post_machine_views));
  
  auto get_cost_of_edge = [&](ParallelComputationGraphEdge const &e) {
    MachineView src_view = pre_machine_views.at(get_src_layer(e));
    MachineView dst_view = post_machine_views.at(get_dst_layer(e));
    ParallelTensorShape tensor_shape = get_parallel_tensor_shape(tr_pcg.full_pcg, 
                                                                 get_parallel_tensor(e));

    return cost_estimator.estimate_cost(tensor_shape, src_view, dst_view);
  };

  // note this is only correct for certain split types, and for others (tensor reuse, etc.) this is 
  // an overapproximation. This should eventually get fixed.
  return sum(transform(edges_across_split, get_cost_of_edge));
}


} // namespace FlexFlow



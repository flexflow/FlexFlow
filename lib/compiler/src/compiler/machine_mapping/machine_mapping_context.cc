#include "compiler/machine_mapping/machine_mapping_context.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/containers/keys.h"
#include "utils/containers/sum.h"

namespace FlexFlow {

std::unordered_set<MachineView> get_allowed_machine_views_for_tensor(MachineMappingContext const &,
                                                                     parallel_tensor_guid_t const &) {
  NOT_IMPLEMENTED();
}

MachineMappingContext make_machine_mapping_context(ParallelComputationGraph const &pcg,
                                                   CostEstimator const &cost_estimator,
                                                   std::function<std::unordered_set<MachineView>(
                                                     ParallelLayerAttrs const &, MachineSpecification const &)> const &allowed_machine_views) {
  NOT_IMPLEMENTED();
}

std::unordered_set<parallel_layer_guid_t> get_transitively_reduced_predecessors(MachineMappingContext const &ctx,
                                                                                parallel_layer_guid_t const &l) {
  NOT_IMPLEMENTED();
}

std::unordered_set<parallel_layer_guid_t> get_transitively_reduced_successors(MachineMappingContext const &ctx,
                                                                              parallel_layer_guid_t const &l) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

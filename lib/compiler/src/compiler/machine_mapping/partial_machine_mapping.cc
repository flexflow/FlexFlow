#include "compiler/machine_mapping/partial_machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_context.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/restrict_keys.h"

namespace FlexFlow {

PartialMachineMapping get_sub_solution(MachineMappingContext const &ctx, 
                                PartialMachineMapping const &partial_solution, 
                                PCGBinarySPDecomposition const &sub_problem) {
  std::unordered_set<parallel_layer_guid_t> sub_solution_layers = 
    flatmap(get_parallel_layers(sub_problem),
            [&](parallel_layer_guid_t l) { 
              return set_union(
                get_transitively_reduced_predecessors(ctx, l),
                get_transitively_reduced_successors(ctx, l));
            });

  return PartialMachineMapping{
    restrict_keys(partial_solution.machine_views, sub_solution_layers),
  };
}

MachineMapping require_complete(MachineMappingContext const &ctx,
                                PartialMachineMapping const &partial_solution) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

#include "compiler/machine_mapping/partial_machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_context.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/map_values.h"

namespace FlexFlow {

PartialMachineMapping get_unconstrained_solution_for_layers(std::unordered_set<parallel_layer_guid_t> const &layers) {
  return PartialMachineMapping{
    generate_map(layers, 
                 [](parallel_layer_guid_t const &) -> std::optional<MachineView> { 
                   return std::nullopt; 
                 }),
  };
}

std::unordered_set<parallel_layer_guid_t> get_all_layers(PartialMachineMapping const &partial_solution, 
                                                         IncludeUnconstrained const &include_unconstrained) {
  std::unordered_set<parallel_layer_guid_t> with_unconstrained = keys(partial_solution.machine_views);

  if (include_unconstrained.raw_bool) {
    return with_unconstrained;
  } else {
    return filter(with_unconstrained,
                  [&](parallel_layer_guid_t const &l) { return partial_solution.machine_views.at(l).has_value(); });
  }
}

std::optional<MachineView> get_machine_view_for_layer(PartialMachineMapping const &partial_solution,
                                                      parallel_layer_guid_t const &layer) {
  return partial_solution.machine_views.at(layer);
}

PartialMachineMapping get_sub_solution(PartialMachineMapping const &partial_solution, 
                                       PCGBinarySPDecomposition const &sub_problem) {

  std::unordered_set<parallel_layer_guid_t> sub_problem_layers = unordered_set_of(get_parallel_layers(sub_problem));

  return PartialMachineMapping{
    restrict_keys(partial_solution.machine_views, sub_problem_layers),
  };
}

PartialMachineMapping with_additional_layer_machine_views(PartialMachineMapping const &partial_solution,
                                                          std::unordered_map<parallel_layer_guid_t, MachineView> const &additional) {
  PartialMachineMapping result = partial_solution;

  for (auto const &[layer, machine_view] : additional) {
    std::optional<MachineView> current_machine_view = result.machine_views.at(layer);

    if (!current_machine_view.has_value()) {
      result.machine_views.at(layer) = machine_view;
    } else {
      if (current_machine_view.value() != machine_view) {
        throw mk_runtime_error(fmt::format("with_additional_layer_machine_views received machine view assignment for layer {} "
                                           "to machine view {}, but that layer is already assigned to machine view {}.", 
                                           layer, machine_view, current_machine_view.value()));
      }
    }
  }

  return result;
}


MachineMapping require_complete_mapping(PartialMachineMapping const &partial_mapping) {
  return MachineMapping{
    map_values(partial_mapping.machine_views, 
               [](std::optional<MachineView> const &mv) { return mv.value(); }),
  };
}

} // namespace FlexFlow

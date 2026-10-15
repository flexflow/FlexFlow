#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "utils/containers/filter.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/filtermap_keys.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/full_binary_tree/binary_tree_path.h"

namespace FlexFlow {

MachineMappingConstraints get_unconstrained_solution_for_layers(
    std::set<BinaryTreePath> const &layers) {
  return MachineMappingConstraints{
      generate_map(layers,
                   [](BinaryTreePath const &) -> std::optional<MachineView> {
                     return std::nullopt;
                   }),
  };
}

std::set<BinaryTreePath>
    get_unconstrained_layers(MachineMappingConstraints const &constraints) {

  return keys(filter_values(
      constraints.machine_views,
      [](std::optional<MachineView> const &mv) { return !mv.has_value(); }));
}

std::set<BinaryTreePath>
    get_constrained_layers(MachineMappingConstraints const &constraints) {

  return keys(filter_values(
      constraints.machine_views,
      [](std::optional<MachineView> const &mv) { return mv.has_value(); }));
}

std::set<BinaryTreePath>
    get_all_layers(MachineMappingConstraints const &partial_solution) {
  return keys(partial_solution.machine_views);
}

std::optional<MachineView> get_machine_view_for_layer(
    MachineMappingConstraints const &partial_solution,
    BinaryTreePath const &layer) {
  return partial_solution.machine_views.at(layer);
}

MachineMappingConstraints
    restrict_to_child(MachineMappingConstraints const &constraints,
                      BinaryTreePathEntry const &prefix) {
  return MachineMappingConstraints{filtermap_keys(
      constraints.machine_views,
      [&](BinaryTreePath const &path) -> std::optional<BinaryTreePath> {
        BinaryTreePathEntry head = binary_tree_path_get_top_level(path);

        if (head == prefix) {
          BinaryTreePath rest = binary_tree_path_get_non_top_level(path);
          return rest;
        } else {
          return std::nullopt;
        }
      })};
}

MachineMappingConstraints
    restrict_to_left_child(MachineMappingConstraints const &c) {
  return restrict_to_child(c, BinaryTreePathEntry::LEFT_CHILD);
}

MachineMappingConstraints
    restrict_to_right_child(MachineMappingConstraints const &c) {
  return restrict_to_child(c, BinaryTreePathEntry::RIGHT_CHILD);
}

MachineMappingConstraints with_additional_constraints(
    MachineMappingConstraints const &constraints,
    ParallelLayerGuidObliviousMachineMapping const &additional) {
  MachineMappingConstraints result = constraints;

  for (auto const &[layer, machine_view] : additional.raw_mapping) {
    std::optional<MachineView> current_machine_view =
        result.machine_views.at(layer);

    if (!current_machine_view.has_value()) {
      result.machine_views.at(layer) = machine_view;
    } else {
      ASSERT(current_machine_view.value() == machine_view,
             fmt::format("with_additional_layer_machine_views received machine "
                         "view assignment for layer {} "
                         "to machine view {}, but that layer is already "
                         "assigned to machine view {}.",
                         layer,
                         machine_view,
                         current_machine_view.value()));
    }
  }

  return result;
}

std::optional<MachineView>
    require_only_root(MachineMappingConstraints const &constraints) {
  ASSERT(keys(constraints.machine_views) == std::set{binary_tree_root_path()},
         fmt::format("require_only_root expected constraints to have only a "
                     "single key (the root path), but received {}",
                     constraints));

  return constraints.machine_views.at(binary_tree_root_path());
}

} // namespace FlexFlow

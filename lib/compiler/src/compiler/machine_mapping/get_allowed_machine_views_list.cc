#include "compiler/machine_mapping/get_allowed_machine_views_list.h"
#include "utils/containers/get_first.h"
#include "utils/containers/set_minus.h"
#include "utils/containers.h"
#include "utils/containers/keys.h"

namespace FlexFlow {

std::vector<std::unordered_map<parallel_layer_guid_t, MachineView>>
    get_allowed_machine_views_list(MachineMappingContext const &context,
                             std::unordered_set<parallel_layer_guid_t> const &layers,
                             MachineSpecification const &resource) {
  if (layers.empty()) {
    return {{}};
  }
  parallel_layer_guid_t curr_layer = get_first(layers);
  std::unordered_set<parallel_layer_guid_t> other_layers = set_minus(layers, {curr_layer});

  std::vector<std::unordered_map<parallel_layer_guid_t, MachineView>> other_machine_views_from_recursion =
      allowed_machine_mappings(context, other_layers, resource);

  std::unordered_set<MachineView> allowed_machine_views_for_curr_layer =
      context.allowed_machine_views(layer, resource);

  std::vector<std::unordered_map<parallel_layer_guid_t, MachineView>> result;

  for (MachineView const &for_curr_node : allowed_machine_views_for_curr_node) {
    for (std::unordered_map<parallel_layer_guid_t, MachineView> const &for_other_nodes :
         other_node_mappings_from_recursion) {
      enumeration.push_back(merge_maps(
          partial, std::unordered_map<parallel_layer_guid_t, MachineView>{{layer, mv}}));
    }
  }
  return result;
}

std::vector<std::unordered_map<parallel_tensor_guid_t, MachineView>>
    get_allowed_src_machine_views_list(MachineMappingContext const &context,
                             std::unordered_set<parallel_tensor_guid_t> const &tensors,
                             MachineSpecification const &resource) {
  std::unordered_set<parallel_layer_guid_t> layers;
  for (parallel_tensor_guid_t const &tensor : tensors) {
    layers.insert(get_source_layer(context.pcg, tensor));
  }

  std::vector<std::unordered_map<parallel_layer_guid_t, MachineView>> machine_views_for_layers_list =
      get_allowed_machine_views_list(context, layers, resource);
  std::vector<std::unordered_map<parallel_tensor_guid_t, MachineView>> result;

  for (std::unordered_map<parallel_layer_guid_t, MachineView> machine_views_for_layers :
       machine_views_for_layers_list) {
    std::unordered_map<parallel_tensor_guid_t, MachineView> machine_views_for_tensors;
    for (parallel_tensor_guid_t const &tensor : tensors) {
      machine_views_for_tensors.emplace(tensor, machine_views_for_layers.at(get_source_layer(context.pcg, v)));
    }
    result.push_back(machine_views_for_tensors);
  }

  return result;
}

}

#include "compiler/machine_mapping/get_allowed_machine_views_list.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "utils/containers.h"
#include "utils/containers/get_first.h"
#include "utils/containers/keys.h"
#include "utils/containers/merge_maps.h"
#include "utils/containers/set_minus.h"

namespace FlexFlow {

std::vector<std::unordered_map<parallel_layer_guid_t, MachineView>>
    get_allowed_machine_views_list(
        MachineMappingContext const &context,
        std::unordered_set<parallel_layer_guid_t> const &layers,
        MachineSpecification const &resource) {
  if (layers.empty()) {
    return {{}};
  }
  parallel_layer_guid_t curr_layer = get_first(layers);
  std::unordered_set<parallel_layer_guid_t> other_layers =
      set_minus(layers, {curr_layer});

  std::vector<std::unordered_map<parallel_layer_guid_t, MachineView>>
      other_machine_views_from_recursion =
          get_allowed_machine_views_list(context, other_layers, resource);

  ParallelLayerAttrs curr_layer_attrs =
      get_parallel_layer_attrs(context.pcg, curr_layer);
  std::unordered_set<MachineView> allowed_machine_views_for_curr_layer =
      context.allowed_machine_views(curr_layer_attrs, resource);

  std::vector<std::unordered_map<parallel_layer_guid_t, MachineView>> result;

  for (MachineView const &for_curr_node :
       allowed_machine_views_for_curr_layer) {
    for (std::unordered_map<parallel_layer_guid_t, MachineView> const
             &for_other_layers : other_machine_views_from_recursion) {
      result.push_back(
          merge_maps(for_other_layers,
                     std::unordered_map<parallel_layer_guid_t, MachineView>{
                         {curr_layer, for_curr_node}}));
    }
  }
  return result;
}

std::vector<std::unordered_map<parallel_tensor_guid_t, MachineView>>
    get_allowed_src_machine_views_list(
        MachineMappingContext const &context,
        std::unordered_set<parallel_tensor_guid_t> const &tensors,
        MachineSpecification const &resource) {
  std::unordered_set<parallel_layer_guid_t> layers;
  for (parallel_tensor_guid_t const &tensor : tensors) {
    layers.insert(get_source_layer(tensor));
  }

  std::vector<std::unordered_map<parallel_layer_guid_t, MachineView>>
      machine_views_for_layers_list =
          get_allowed_machine_views_list(context, layers, resource);

  std::vector<std::unordered_map<parallel_tensor_guid_t, MachineView>> result;

  for (std::unordered_map<parallel_layer_guid_t, MachineView>
           machine_views_for_layers : machine_views_for_layers_list) {
    std::unordered_map<parallel_tensor_guid_t, MachineView>
        machine_views_for_tensors;
    for (parallel_tensor_guid_t const &tensor : tensors) {
      machine_views_for_tensors.emplace(
          tensor, machine_views_for_layers.at(get_source_layer(tensor)));
    }
    result.push_back(machine_views_for_tensors);
  }

  return result;
}

} // namespace FlexFlow

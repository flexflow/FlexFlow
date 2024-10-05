#include "compiler/machine_mapping/get_machine_resource_splits.h"
#include "utils/hash/pair.h"

namespace FlexFlow {

std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
    get_machine_resource_splits(MachineSpecification const &resource) {
  std::unordered_set<std::pair<MachineSpecification, MachineSpecification>>
      result;

  for (int i = 1; i < resource.num_nodes; i *= 2) {
    MachineSpecification sub_resource1 = resource;
    MachineSpecification sub_resource2 = resource;
    sub_resource1.num_nodes = i;
    sub_resource2.num_nodes = resource.num_nodes - i;
    result.insert(std::make_pair(sub_resource1, sub_resource2));
    result.insert(std::make_pair(sub_resource2, sub_resource1));
  }

  for (int i = 1; i < resource.num_gpus_per_node; i *= 2) {
    MachineSpecification sub_resource1 = resource;
    MachineSpecification sub_resource2 = resource;
    sub_resource1.num_gpus_per_node = i;
    sub_resource2.num_gpus_per_node = resource.num_gpus_per_node - i;
    result.insert(std::make_pair(sub_resource1, sub_resource2));
    result.insert(std::make_pair(sub_resource2, sub_resource1));
  }

  return result;
}

} // namespace FlexFlow

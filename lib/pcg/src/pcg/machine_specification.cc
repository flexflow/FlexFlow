#include "pcg/machine_specification.h"
#include "pcg/device_id.h"
#include "utils/exception.h"
namespace FlexFlow {

int get_num_gpus(MachineSpecification const &ms) {
  return ms.num_nodes * ms.num_gpus_per_node;
}
int get_num_cpus(MachineSpecification const &ms) {
  return ms.num_nodes * ms.num_cpus_per_node;
}
int get_num_devices(MachineSpecification const &ms,
                    DeviceType const &device_type) {
  switch (device_type) {
    case DeviceType::GPU:
      return get_num_gpus(ms);
    case DeviceType::CPU:
      return get_num_cpus(ms);
    default:
      throw mk_runtime_error("Unknown DeviceType {}", device_type);
  }
}

int get_num_devices_per_node(MachineSpecification const &ms,
                             DeviceType const &device_type) {
  switch (device_type) {
    case DeviceType::GPU:
      return ms.num_gpus_per_node;
    case DeviceType::CPU:
      return ms.num_cpus_per_node;
    default:
      throw mk_runtime_error("Unknown DeviceType {}", device_type);
  }
}
bool is_valid_machine_space_coordinate(MachineSpecification const &ms,
                                       MachineSpaceCoordinate const &coord) {
  return (coord.node_idx < ms.num_nodes) &&
         (coord.device_idx < get_num_devices_per_node(ms, coord.device_type));
}

device_id_t get_device_id(MachineSpecification const &ms,
                          MachineSpaceCoordinate const &coord) {
  if (!is_valid_machine_space_coordinate(ms, coord)) {
    throw mk_runtime_error(fmt::format(
        "Invalid coordinate {} for machine specification {}", ms, coord));
  }
  int raw_idx =
      coord.node_idx * get_num_devices_per_node(ms, coord.device_type) +
      coord.device_idx;
  return device_id_from_index(raw_idx, coord.device_type);
}

} // namespace FlexFlow

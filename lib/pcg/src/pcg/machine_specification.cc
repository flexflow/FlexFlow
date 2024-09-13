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
bool is_valid_machine_specification_coordinates(
    MachineSpecification const &ms,
    MachineSpecificationCoordinates const &coords) {
  return (coords.inter < ms.num_nodes) &&
         (coords.intra < get_num_devices_per_node(ms, coords.device_type));
}

device_id_t get_device_id(MachineSpecification const &ms,
                          MachineSpecificationCoordinates const &coords) {
  assert(is_valid_machine_specification_coordinates(ms, coords));
  int raw_idx =
      coords.inter * get_num_devices_per_node(ms, coords.device_type) +
      coords.intra;
  return device_id_from_index(raw_idx, coords.device_type);
}

} // namespace FlexFlow

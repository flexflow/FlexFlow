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
} // namespace FlexFlow

#include "pcg/device_id.h"
#include "utils/exception.h"
#include <cassert>

namespace FlexFlow {

device_id_t operator+(device_id_t, size_t) {
  NOT_IMPLEMENTED();
}

DeviceType get_device_type(device_id_t const &device_id) {
  if (device_id.has<gpu_id_t>()) {
    return DeviceType::GPU;
  } else {
    assert(device_id.has<cpu_id_t>());
    return DeviceType::CPU;
  }
}

gpu_id_t unwrap_gpu(device_id_t device_id) {
  return device_id.get<gpu_id_t>();
}

cpu_id_t unwrap_cpu(device_id_t device_id) {
  return device_id.get<cpu_id_t>();
}

int get_raw_id(device_id_t device_id) {
  switch (get_device_type(device_id)) {
    case DeviceType::GPU:
      return unwrap_gpu(device_id).gpu_index;
    case DeviceType::CPU:
      return unwrap_cpu(device_id).cpu_index;
    default:
      throw mk_runtime_error(
          fmt::format("Unsupported DeviceType {} for device_id_t {}",
                      get_device_type(device_id),
                      device_id));
  }
}

device_id_t device_id_from_index(int, DeviceType) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

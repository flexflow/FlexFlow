#include "pcg/device_id.h"
#include "utils/exception.h"
#include <cassert>

namespace FlexFlow {

DeviceType get_device_type(device_id_t const &id) {
  if (std::holds_alternative<gpu_id_t>(id)) {
    return DeviceType::GPU;
  } else {
    assert(std::holds_alternative<cpu_id_t>(id));
    return DeviceType::CPU;
  }
}

//Most likely not the best way to do it.
device_id_t operator+(device_id_t device, size_t increment) {
  if (get_device_type(device) == DeviceType::GPU) {
    gpu_id_t val = std::get<gpu_id_t>(device);
    return val+increment;
  }
  else {
    cpu_id_t val = std::get<cpu_id_t>(device);
    return val+increment;
  }
}

} // namespace FlexFlow

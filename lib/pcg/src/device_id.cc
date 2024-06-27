#include "pcg/device_id.h"
#include "utils/exception.h"
#include <cassert>
#include <variant>

namespace FlexFlow {

DeviceType get_device_type(device_id_t const &id) {
  if (std::holds_alternative<gpu_id_t>(id)) {
    return DeviceType::GPU;
  } else {
    assert(std::holds_alternative<cpu_id_t>(id));
    return DeviceType::CPU;
  }
}

// Most likely not the best way to do it.
device_id_t operator+(device_id_t device, size_t increment) {
  if (std::holds_alternative<gpu_id_t>(device)) {
    gpu_id_t gpu_id = std::get<gpu_id_t>(device);
    int new_value = static_cast<int>(gpu_id) + static_cast<int>(increment);
    return gpu_id_t(new_value);
  } else {
    assert((std::holds_alternative<cpu_id_t>(device)));
    cpu_id_t cpu_id = std::get<cpu_id_t>(device);
    int new_value = static_cast<int>(cpu_id) + static_cast<int>(increment);
    return cpu_id_t(new_value);
  }
}

} // namespace FlexFlow

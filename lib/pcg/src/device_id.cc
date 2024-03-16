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

device_id_t operator+(device_id_t, size_t) {
  NOT_IMPLEMENTED();
}
} // namespace FlexFlow

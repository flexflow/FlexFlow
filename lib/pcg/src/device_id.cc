#include "pcg/device_id.h"
#include <cassert>

namespace FlexFlow {

device_id_t operator+(device_id_t, size_t) {
  NOT_IMPLEMENTED();
}

DeviceType get_device_type(device_id_t const &id) {
  if (holds_alternative<gpu_id_t>(id)) {
    return DeviceType::GPU;
  } else {
    assert(holds_alternative<cpu_id_t>(id));
    return DeviceType::CPU;
  }
}

} // namespace FlexFlow

#include "pcg/device_id.h"
#include <cassert>

namespace FlexFlow {

DeviceType get_device_type(device_id_t const &id) {
  if (holds_alternative<gpu_id_t>(id)) {
    return DeviceType::GPU;
  } else {
    assert(holds_alternative<cpu_id_t>(id));
    return DeviceType::CPU;
  }
}

} // namespace FlexFlow

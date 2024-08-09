#include "local-execution/per_device_state.h"
#include "utils/overload.h"

namespace FlexFlow {

PerDeviceState get_device_state_from_device_specific(
    DeviceSpecificDeviceStates const &device_specific, size_t device_idx) {
  return device_specific.visit<PerDeviceState>(
      [&](auto const &x) { return PerDeviceState{*(x.get(device_idx))}; });
}

} // namespace FlexFlow

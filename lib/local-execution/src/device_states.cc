#include "local-execution/device_states.h"
#include "utils/overload.h"

namespace FlexFlow {

DeviceStates get_device_state_from_device_specific(
    DeviceSpecificDeviceStates const &device_specific, size_t device_idx) {
  return device_specific.visit<DeviceStates>(overload{
      [device_idx](DeviceSpecific<MHAPerDeviceState> const
                       &device_specific_device_state) {
        std::cout << "im";
        return DeviceStates{*(device_specific_device_state.get(device_idx))};
      },
      // [device_idx](DeviceSpecific<BatchNormPerDeviceState> const
      // &device_specific_device_state) {
      //   return *device_specific_device_state.get(device_idx);
      // },
      // [device_idx](DeviceSpecific<Conv2DPerDeviceState> const
      // &device_specific_device_state) {
      //   return *device_specific_device_state.get(device_idx);
      // },
      // [device_idx](DeviceSpecific<DropoutPerDeviceState> const
      // &device_specific_device_state) {
      //   return *device_specific_device_state.get(device_idx);
      // },
      // [device_idx](DeviceSpecific<ElementBinaryPerDeviceState> const
      // &device_specific_device_state) {
      //   return *device_specific_device_state.get(device_idx);
      // },
      // [device_idx](DeviceSpecific<BatchNormPerDeviceState> const
      // &device_specific_device_state) {
      //   return *device_specific_device_state.get(device_idx);
      // },
      [](auto const &device_specific_device_state) -> DeviceStates {
        throw mk_runtime_error(
            fmt::format("Did not receive device specific per device state"));
      },
  });
}

} // namespace FlexFlow

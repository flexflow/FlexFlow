#include "test_utils.h"

namespace FlexFlow {

PerDeviceFFHandle get_mock_per_device_ff_handle() {
  return {nullptr, nullptr, nullptr, 0, false};
}

} // namespace FlexFlow

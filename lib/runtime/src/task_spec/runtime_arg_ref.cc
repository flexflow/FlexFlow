#include "runtime_arg_ref.h"
#include "device_specific_arg.h"

namespace FlexFlow {

RuntimeArgRef<ProfilingSettings> profiling_settings() {
  return {RuntimeArgRefType::PROFILING_SETTINGS};
}

RuntimeArgRef<DeviceSpecificArg<PerDeviceFFHandle>> ff_handle() {
  return {RuntimeArgRefType::FF_HANDLE};
}

} // namespace FlexFlow

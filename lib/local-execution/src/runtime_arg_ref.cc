#include "local-execution/runtime_arg_ref.h"
#include "local-execution/device_specific.h"

namespace FlexFlow {

RuntimeArgRef<ProfilingSettings> profiling_settings() {
  return {RuntimeArgRefType::PROFILING_SETTINGS};
}

RuntimeArgRef<DeviceSpecific<PerDeviceFFHandle>> ff_handle() {
  return {RuntimeArgRefType::FF_HANDLE};
}

RuntimeArgRef<FFIterationConfig> iteration_config() {
  return {RuntimeArgRefType::FF_ITERATION_CONFIG};
}

} // namespace FlexFlow

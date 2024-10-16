#include "local-execution/runtime_arg_ref.h"
#include "local-execution/device_specific.h"

namespace FlexFlow {

std::string to_string(RuntimeArgRefType const &runtime_arg_ref_type) {
  switch (runtime_arg_ref_type) {
    case RuntimeArgRefType::FF_HANDLE:
      return "FF_HANDLE";
    case RuntimeArgRefType::PROFILING_SETTINGS:
      return "PROFILING_SETTINGS";
    case RuntimeArgRefType::FF_ITERATION_CONFIG:
      return "FF_ITERATION_CONFIG";
    default:
      return "Unknown";
  }
}

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

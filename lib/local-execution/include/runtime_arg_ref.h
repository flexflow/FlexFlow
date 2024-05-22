#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H

#include "arg_ref.h"
#include "config.h"
#include "device_specific.h"

namespace FlexFlow {

enum class RuntimeArgRefType {
  FF_HANDLE,
  PROFILING_SETTINGS,
  FF_ITERATION_CONFIG
};

template <typename T>
using RuntimeArgRef = ArgRef<RuntimeArgRefType, T>;

using RuntimeArgRefSpec = ArgRefSpec<RuntimeArgRefType>;

RuntimeArgRef<ProfilingSettings> profiling_settings() {
  return {RuntimeArgRefType::PROFILING_SETTINGS};
}

RuntimeArgRef<DeviceSpecific<PerDeviceFFHandle>> ff_handle() {
  return {RuntimeArgRefType::FF_HANDLE};
}

RuntimeArgRef<DeviceSpecific<PerDeviceFFHandle>> ff_handle() {
  return {RuntimeArgRefType::FF_ITERATION_CONFIG};
}

} // namespace FlexFlow

#endif

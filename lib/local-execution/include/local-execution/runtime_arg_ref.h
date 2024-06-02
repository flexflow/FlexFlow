#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H

#include "local-execution/arg_ref.h"
#include "local-execution/config.h"
#include "local-execution/device_specific.h"

namespace FlexFlow {

enum class RuntimeArgRefType {
  FF_HANDLE,
  PROFILING_SETTINGS,
  FF_ITERATION_CONFIG
};

template <typename T>
using RuntimeArgRef = ArgRef<RuntimeArgRefType, T>;

using RuntimeArgRefSpec = ArgRefSpec<RuntimeArgRefType>;

RuntimeArgRef<ProfilingSettings> profiling_settings();
RuntimeArgRef<DeviceSpecific<PerDeviceFFHandle>> ff_handle();
RuntimeArgRef<DeviceSpecific<PerDeviceFFHandle>> iteration_config();

} // namespace FlexFlow

#endif

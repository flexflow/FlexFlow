#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H

#include "arg_ref.h"
#include "device_specific.h"
#include "runtime/config.h"

namespace FlexFlow {

enum class RuntimeArgRefType { FF_HANDLE, PROFILING_SETTINGS };

template <typename T>
using RuntimeArgRef = ArgRef<RuntimeArgRefType, T>;

using RuntimeArgRefSpec = ArgRefSpec<RuntimeArgRefType>;

RuntimeArgRef<ProfilingSettings> profiling_settings();
RuntimeArgRef<DeviceSpecific<PerDeviceFFHandle>> ff_handle();
RuntimeArgRef<FFIterationConfig> iteration_config();

} // namespace FlexFlow

#endif
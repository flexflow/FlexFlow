#ifndef _FLEXFLOW_LOCAL_EXECUTION_RUNTIME_ARG_CONFIG_H
#define _FLEXFLOW_LOCAL_EXECUTION_RUNTIME_ARG_CONFIG_H

#include "kernels/ff_handle.h"
#include "local-execution/device_specific.h"
#include "local-execution/profiling.h"

namespace FlexFlow {

struct RuntimeArgConfig {
public:
  DeviceSpecific<PerDeviceFFHandle> ff_handle;
  EnableProfiling enable_profiling;
  ProfilingSettings profiling_settings;
};

} // namespace FlexFlow

#endif

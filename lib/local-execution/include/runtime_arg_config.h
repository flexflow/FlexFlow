#ifndef _FLEXFLOW_LOCAL_EXECUTION_RUNTIME_ARG_CONFIG_H
#define _FLEXFLOW_LOCAL_EXECUTION_RUNTIME_ARG_CONFIG_H

#include "profiling.h"
#include "device_specific.h"
#include "kernels/ff_handle.h"

namespace FlexFlow {

struct RuntimeArgConfig {
public:
  DeviceSpecific<PerDeviceFFHandle> ff_handle;
  EnableProfiling enable_profiling;
  ProfilingSettings profiling_settings;
};

}

#endif

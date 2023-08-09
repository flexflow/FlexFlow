#include "runtime_arg_ref.h"

namespace FlexFlow {

RuntimeArgRef<ProfilingSettings> profiling_settings() {
  return {RuntimeArgRefType::PROFILING_SETTINGS};
}

RuntimeArgRef<PerDeviceFFHandle> ff_handle() {
  return {RuntimeArgRefType::FF_HANDLE};
}

} // namespace FlexFlow

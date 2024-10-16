#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H

#include "local-execution/arg_ref.h"
#include "local-execution/config.h"
#include "local-execution/device_specific.h"
#include "local-execution/profiling.h"
#include "utils/fmt.h"
#include "utils/type_index.h"

namespace FlexFlow {

enum class RuntimeArgRefType {
  FF_HANDLE,
  PROFILING_SETTINGS,
  FF_ITERATION_CONFIG
};

std::string to_string(RuntimeArgRefType const &);

template <typename T>
using RuntimeArgRef = ArgRef<RuntimeArgRefType, T>;

using RuntimeArgRefSpec = ArgRefSpec<RuntimeArgRefType>;

RuntimeArgRef<ProfilingSettings> profiling_settings();
RuntimeArgRef<DeviceSpecific<PerDeviceFFHandle>> ff_handle();
RuntimeArgRef<FFIterationConfig> iteration_config();

// std::string format_as(RuntimeArgRefSpec const & x) {
//   std::ostringstream oss;
//   oss << "<RuntimeArgRefSpec";
//   oss << " type_idx=" << x.get_type_index().name();
//   oss << ">";
//   return oss.str();
// }

// std::ostream &operator<<(std::ostream & s, RuntimeArgRefSpec const & x) {
//   return (s << fmt::to_string(x));
// }

} // namespace FlexFlow

#endif

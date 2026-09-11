#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PROFILING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PROFILING_H

#include "kernels/profiling.h"
#include "utils/optional.h"
#include <spdlog/spdlog.h>

namespace FlexFlow {

template <typename F, typename... Ts, typename Str>
std::optional<milliseconds_t>
    profile(F const &f,
            std::optional<ProfilingSettings> const &profiling,
            DeviceType device_type,
            Str s,
            Ts &&...ts) {
  if (!profiling.has_value()) {
    f(get_stream_for_device_type(device_type), std::forward<Ts>(ts)...);
    return std::nullopt;
  } else {
    ProfilingSettings settings = assert_unwrap(profiling);
    milliseconds_t elapsed = profiling_wrapper<F, Ts...>(
        f, profiling.value(), device_type, std::forward<Ts>(ts)...);
    spdlog::debug(s, elapsed);
    return elapsed;
  }
}

} // namespace FlexFlow

#endif

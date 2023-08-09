#ifndef _FLEXFLOW_RUNTIME_SRC_PROFILING_H
#define _FLEXFLOW_RUNTIME_SRC_PROFILING_H

#include "kernels/profiling.h"
#include "legion.h"
#include "loggers.h"

namespace FlexFlow {

enum class EnableProfiling { YES, NO };

template <typename F, typename... Ts, typename Str>
optional<float>
    profile(F const &f, ProfilingSettings profiling, Str s, Ts &&...ts) {
  optional<float> elapsed =
      profiling_wrapper<F, Ts...>(f, profiling, std::forward<Ts>(ts)...);
  if (elapsed.has_value()) {
    log_profile.debug(s, elapsed.value());
  }
  return elapsed;
}

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_RUNTIME_SRC_PROFILING_H
#define _FLEXFLOW_RUNTIME_SRC_PROFILING_H

#include "kernels/profiling.h"

namespace FlexFlow {

enum class EnableProfiling { YES, NO };

template <typename F, typename... Ts, typename Str>
std::optional<float>
    profile(F const &f, ProfilingSettings profiling, Str s, Ts &&...ts) {
  std::optional<float> elapsed =
      profiling_wrapper<F, Ts...>(f, profiling, std::forward<Ts>(ts)...);
  // TODO -- local logger?
  // if (elapsed.has_value()) {
  //   log_profile.debug(s, elapsed.value());
  // }
  return elapsed;
}

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_RUNTIME_SRC_PROFILING_H
#define _FLEXFLOW_RUNTIME_SRC_PROFILING_H

#include "legion.h"
#include "kernels/profiling.h"

namespace FlexFlow {

extern LegionRuntime::Logger::Category log_profile;

enum class EnableProfiling {
  YES,
  NO
};

template <typename F, typename ...Ts, typename Str>
void profile(F const &f, EnableProfiling profiling, Str s, Ts &&...ts) {
  optional<float> elapsed = profiling_wrapper<F, Ts...>(f, profiling == EnableProfiling::YES, std::forward<Ts>(ts)...);
  if (elapsed.has_value()) {
    log_profile.debug(s, elapsed.value());
  }
}


}

#endif

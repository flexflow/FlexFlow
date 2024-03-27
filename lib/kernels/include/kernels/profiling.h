#ifndef _FLEXFLOW_KERNELS_PROFILING_H
#define _FLEXFLOW_KERNELS_PROFILING_H

#include "device.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ProfilingSettings : public use_visitable_cmp<ProfilingSettings> {
public:
  ProfilingSettings() = delete;
  ProfilingSettings(int warmup_iters, int measure_iters);

public:
  int warmup_iters;
  int measure_iters;
};

template <typename F, typename... Ts>
std::optional<float>
    profiling_wrapper(F const &f, bool enable_profiling, Ts &&...ts) {
  if (enable_profiling) {
    ProfilingSettings settings = {0, 1};
    return profiling_wrapper<F, Ts...>(f, settings, std::forward<Ts>(ts)...);
  } else {
    ffStream_t stream;
    checkCUDA(get_legion_stream(&stream));
    f(stream, std::forward<Ts>(ts)...);
    return std::nullopt;
  }
}

template <typename F, typename... Ts>
std::optional<float> profiling_wrapper(F const &f,
                                  ProfilingSettings const &settings,
                                  Ts &&...ts) {
  ffStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  ffEvent_t t_start, t_end;
  checkCUDA(ffEventCreate(&t_start));
  checkCUDA(ffEventCreate(&t_end));

  for (int i = 0; i < settings.warmup_iters + settings.measure_iters; i++) {
    if (i == settings.warmup_iters) {
      checkCUDA(ffEventRecord(t_start, stream));
    }
    f(stream, std::forward<Ts>(ts)...);
  }

  float elapsed = 0;
  checkCUDA(ffEventRecord(t_end, stream));
  checkCUDA(ffEventSynchronize(t_end));
  checkCUDA(ffEventElapsedTime(&elapsed, t_start, t_end));
  checkCUDA(ffEventDestroy(t_start));
  checkCUDA(ffEventDestroy(t_end));
  return elapsed;
}

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ProfilingSettings, warmup_iters, measure_iters);
MAKE_VISIT_HASHABLE(::FlexFlow::ProfilingSettings);

#endif

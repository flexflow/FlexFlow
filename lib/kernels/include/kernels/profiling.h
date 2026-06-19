#ifndef _FLEXFLOW_KERNELS_PROFILING_H
#define _FLEXFLOW_KERNELS_PROFILING_H

#include "kernels/device.h"
#include "kernels/device_stream_t.h"
#include "kernels/profiling_settings.dtg.h"
#include "pcg/device_type.dtg.h"
#include "utils/units/milliseconds_t.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename F, typename... Ts>
std::optional<milliseconds_t> profiling_wrapper(F const &f,
                                                bool enable_profiling,
                                                DeviceType device_type,
                                                Ts &&...ts) {
  if (enable_profiling) {
    ProfilingSettings settings = ProfilingSettings{
        /*warmup_iters=*/0_n,
        /*measure_iters=*/1_p,
    };
    return profiling_wrapper<F, Ts...>(f, settings, std::forward<Ts>(ts)...);
  } else {
    f(get_stream_for_device_type(device_type), std::forward<Ts>(ts)...);
    return std::nullopt;
  }
}

template <typename F, typename... Ts>
std::optional<milliseconds_t>
    profiling_wrapper(F const &f,
                      ProfilingSettings const &settings,
                      DeviceType device_type,
                      Ts &&...ts) {
  if (settings.measure_iters.int_from_positive_int() <= 0) {
    return std::nullopt;
  }

  if (device_type == DeviceType::GPU) {
    return gpu_profiling_wrapper(f, settings, std::forward<Ts>(ts)...);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return cpu_profiling_wrapper(f, settings, std::forward<Ts>(ts)...);
  }
}

template <typename F, typename... Ts>
milliseconds_t cpu_profiling_wrapper(F const &f,
                                     ProfilingSettings const &settings,
                                     Ts &&...ts) {
  ASSERT(settings.measure_iters.int_from_positive_int() > 0);

  device_stream_t stream = get_cpu_device_stream();

  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  std::optional<TimePoint> start = std::nullopt;
  std::optional<TimePoint> end = std::nullopt;

  for (int i = 0; i < settings.warmup_iters.int_from_nonnegative_int() +
                          settings.measure_iters.int_from_positive_int();
       i++) {
    if (i == settings.warmup_iters.int_from_nonnegative_int()) {
      start = std::chrono::steady_clock::now();
    }
    f(stream, std::forward<Ts>(ts)...);
  }
  end = std::chrono::steady_clock::now();

  std::chrono::duration<double, std::milli> avg_duration =
      (end.value() - start.value()) /
      settings.measure_iters.int_from_positive_int();

  return milliseconds_t{
      static_cast<float>(avg_duration.count()),
  };
}

template <typename F, typename... Ts>
milliseconds_t gpu_profiling_wrapper(F const &f,
                                     ProfilingSettings const &settings,
                                     Ts &&...ts) {
  ASSERT(settings.measure_iters.int_from_positive_int() > 0);

  device_stream_t stream = get_gpu_device_stream();

  ffEvent_t t_start, t_end;
  checkCUDA(ffEventCreate(&t_start));
  checkCUDA(ffEventCreate(&t_end));

  for (int i = 0; i < settings.warmup_iters.int_from_nonnegative_int() +
                          settings.measure_iters.int_from_positive_int();
       i++) {
    if (i == settings.warmup_iters.int_from_nonnegative_int()) {
      checkCUDA(ffEventRecord(t_start, stream.require_gpu()));
    }
    f(stream, std::forward<Ts>(ts)...);
  }

  float elapsed = 0;
  checkCUDA(ffEventRecord(t_end, stream.require_gpu()));
  checkCUDA(ffEventSynchronize(t_end));
  checkCUDA(ffEventElapsedTime(&elapsed, t_start, t_end));
  checkCUDA(ffEventDestroy(t_start));
  checkCUDA(ffEventDestroy(t_end));
  return milliseconds_t{
      elapsed / settings.measure_iters.int_from_positive_int(),
  };
}

} // namespace FlexFlow

#endif

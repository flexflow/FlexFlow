#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_PERF_METRICS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_PERF_METRICS_H

#include "utils/fmt.h"
#include "utils/optional.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct PerfMetrics : public use_visitable_cmp<PerfMetrics> {
  PerfMetrics() = delete;
  PerfMetrics(double start_time);
  PerfMetrics(int train_all,
              optional<int> train_correct,
              optional<float> cce_loss,
              optional<float> sparse_cce_loss,
              optional<float> mse_loss,
              optional<float> rmse_loss,
              optional<float> mae_loss,
              double start_time_micro,
              double current_time_micro);

  int train_all = 0;                  // measure_accuracy_denominator
  optional<int> train_correct = 0;    // measure_accuracy numerator
  optional<float> cce_loss = nullopt; // measure_categorical_crossentropy
  optional<float> sparse_cce_loss =
      0.0f;                         // measure_sparse_categorical_crossentropy
  optional<float> mse_loss = 0.0f;  // measure_mean_squared_error
  optional<float> rmse_loss = 0.0f; // measure_root_mean_squared_error
  optional<float> mae_loss = 0.0f;  // measure_mean_absolute_error
  double start_time;
  double current_time;
};

float get_throughput(PerfMetrics const &);
float get_accuracy(PerfMetrics const &);

PerfMetrics update(PerfMetrics const &, PerfMetrics const &);
PerfMetrics apply_scale(PerfMetrics const &, float scale);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::PerfMetrics,
                 train_all,
                 train_correct,
                 cce_loss,
                 sparse_cce_loss,
                 mse_loss,
                 rmse_loss,
                 mae_loss,
                 start_time);

namespace fmt {

template <>
struct formatter<::FlexFlow::PerfMetrics> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::PerfMetrics const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    auto out = fmt::memory_buffer();
    fmt::format_to(std::back_inserter(out), "PerfMetrics[");
    if (m.train_correct.has_value()) {
      fmt::format_to(std::back_inserter(out),
                     " accuracy={:.2f}%",
                     100.0 * get_accuracy(m));
    }
    if (m.cce_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " cce={:.2f}", m.cce_loss.value());
    }
    if (m.sparse_cce_loss.has_value()) {
      fmt::format_to(std::back_inserter(out),
                     " sparse_cce={:.2f}",
                     m.sparse_cce_loss.value());
    }
    if (m.mse_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " mse={:.2f}", m.mse_loss.value());
    }
    if (m.rmse_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " rmse={:.2f}", m.rmse_loss.value());
    }
    if (m.mae_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " mae={:.2f}", m.mae_loss.value());
    }
    fmt::format_to(
        std::back_inserter(out), "throughput={:.2f}", get_throughput(m));
    return formatter<std::string>::format(fmt::to_string(out), ctx);
  }
};

} // namespace fmt

#endif

#include "doctest.h"
#include "kernels/perf_metrics.h"

using namespace FlexFlow;

TEST_CASE("PerfMetrics Tests") {

  SUBCASE("Constructor and basic properties") {
    double start_time = 100.0;
    PerfMetrics metrics(start_time);

    CHECK(metrics.start_time == start_time);
    CHECK(metrics.current_time == start_time);
    CHECK(metrics.train_all == 0);
  }

  SUBCASE("Throughput non-negative") {
    PerfMetrics m(10, 5, 0.5, 0.5, 0.5, 0.5, 0.5, 100.0, 200.0);
    CHECK(get_throughput(m) >= 0);
  }

  SUBCASE("Accuracy between 0 and 1") {
    PerfMetrics m(10, 5, 0.5, 0.5, 0.5, 0.5, 0.5, 100.0, 200.0);
    float accuracy = get_accuracy(m);
    CHECK(accuracy >= 0.0f);
    CHECK(accuracy <= 1.0f);
  }

  SUBCASE("Update maintains non-negative values") {
    PerfMetrics lhs(10, 5, 0.5, 0.5, 0.5, 0.5, 0.5, 100.0, 200.0);
    PerfMetrics rhs(5, 3, 0.2, 0.2, 0.2, 0.2, 0.2, 200.0, 300.0);

    auto result = update(lhs, rhs);
    CHECK(result.train_all == 15);
    CHECK(result.train_correct.value() == 8);
  }

  SUBCASE("Scale values correctly") {
    PerfMetrics pm(10, 5, 0.5, 0.5, 0.5, 0.5, 0.5, 100.0, 200.0);
    float scale = 2.0f;

    auto result = apply_scale(pm, scale);
    CHECK(result.cce_loss.value() == doctest::Approx(1.0f));
    CHECK(result.sparse_cce_loss.value() == doctest::Approx(1.0f));
    CHECK(result.mse_loss.value() == doctest::Approx(1.0f));
    CHECK(result.rmse_loss.value() == doctest::Approx(1.0f));
    CHECK(result.mae_loss.value() == doctest::Approx(1.0f));
  }
}

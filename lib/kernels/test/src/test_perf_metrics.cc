#include "doctest.h"
#include "kernels/perf_metrics.h"
#include <random>

using namespace FlexFlow;

// Helper function to generate random values for PerfMetrics
PerfMetrics randomPerfMetrics() {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<> dis(0.0, 1.0);

  PerfMetrics metrics{0};
  metrics.train_all = dis(gen);
  metrics.train_correct = dis(gen);
  metrics.cce_loss = dis(gen);
  metrics.sparse_cce_loss = dis(gen);
  metrics.mse_loss = dis(gen);
  metrics.rmse_loss = dis(gen);
  metrics.mae_loss = dis(gen);
  metrics.start_time = dis(gen);
  metrics.current_time = dis(gen);

  return metrics;
}

TEST_CASE("PerfMetricsTests1 " ) {

  SUBCASE("Throughput non-negative") {
    auto m = randomPerfMetrics();
    CHECK(get_throughput(m) >= 0);
  }

  SUBCASE("Accuracy between 0 and 1") {
    auto m = randomPerfMetrics();
    float accuracy = get_accuracy(m);
    CHECK(accuracy >= 0.0f);
    CHECK(accuracy <= 1.0f);
  }

  SUBCASE("Update maintains non-negative values") {
    auto lhs = randomPerfMetrics();
    auto rhs = randomPerfMetrics();
    auto result = update(lhs, rhs);
    CHECK(result.train_all >= 0);
    // Add other assertions for other fields...
  }
}


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


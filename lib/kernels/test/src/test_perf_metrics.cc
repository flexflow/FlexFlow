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

TEST_CASE("PerfMetricsTests") {

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

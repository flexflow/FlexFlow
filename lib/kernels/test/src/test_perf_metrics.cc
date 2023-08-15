// #include "kernels/perf_metrics.h"
// #include "rapidcheck.h"

// using namespace FlexFlow;

// // 1. Generator for PerfMetrics
// namespace rc {

// template <>
// struct Arbitrary<PerfMetrics> {
//   static Gen<PerfMetrics> arbitrary() {
//     return gen::build<PerfMetrics>(gen::set(&PerfMetrics::train_all),
//                                    gen::set(&PerfMetrics::train_correct),
//                                    gen::set(&PerfMetrics::cce_loss),
//                                    gen::set(&PerfMetrics::sparse_cce_loss),
//                                    gen::set(&PerfMetrics::mse_loss),
//                                    gen::set(&PerfMetrics::rmse_loss),
//                                    gen::set(&PerfMetrics::mae_loss),
//                                    gen::set(&PerfMetrics::start_time),
//                                    gen::set(&PerfMetrics::current_time));
//   }
// };

// } // namespace rc

// // 2. Properties for PerfMetrics

// RC_GTEST_PROP(PerfMetricsTests, "Throughput non-negative", ()) {
//   auto const m = *rc::gen::arbitrary<PerfMetrics>();
//   RC_ASSERT(get_throughput(m) >= 0);
// }

// RC_GTEST_PROP(PerfMetricsTests, "Accuracy between 0 and 1", ()) {
//   auto const m = *rc::gen::arbitrary<PerfMetrics>();
//   float accuracy = get_accuracy(m);
//   RC_ASSERT(accuracy >= 0.0f && accuracy <= 1.0f);
// }

// RC_GTEST_PROP(PerfMetricsTests, "Update maintains non-negative values", ()) {
//   auto const lhs = *rc::gen::arbitrary<PerfMetrics>();
//   auto const rhs = *rc::gen::arbitrary<PerfMetrics>();
//   auto const result = update(lhs, rhs);
//   RC_ASSERT(result.train_all >= 0);
//   // Add other assertions for other fields...
// }

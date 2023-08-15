#include "kernels/profiling.h"
#include "rapidcheck.h"

using namespace FlexFlow;

// Generator for ProfilingSettings
namespace rc {
template <>
struct Arbitrary<ProfilingSettings> {
  static Gen<ProfilingSettings> arbitrary() {
    return gen::build<ProfilingSettings>(
        gen::inRange(0, 10), // for simplicity, let's use small ranges
        gen::inRange(0, 10));
  }
};
} // namespace rc

// Properties for ProfilingSettings

// Check if creating and using ProfilingSettings works as expected
RC_GTEST_PROP(ProfilingSettingsTests,
              "Check valid warmup and measure iters",
              ()) {
  auto settings = *rc::gen::arbitrary<ProfilingSettings>();
  RC_ASSERT(settings.warmup_iters >= 0 && settings.warmup_iters <= 10);
  RC_ASSERT(settings.measure_iters >= 0 && settings.measure_iters <= 10);
}

// Properties for profiling_wrapper
// Note: You will need to provide or mock a function to test the wrapper

RC_GTEST_PROP(ProfilingWrapperTests,
              "Profiling time should be non-negative",
              ()) {
  auto settings = *rc::gen::arbitrary<ProfilingSettings>();
  auto mockFunc = [](ffStream_t stream) { /*mock function body*/ };
  auto result = profiling_wrapper(mockFunc, settings);
  if (result.has_value()) {
    RC_ASSERT(result.value() >= 0.0f);
  }
}
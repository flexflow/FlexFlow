#include "pcg/initializers/uniform_initializer_attrs.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Arbitrary<UniformInitializerAttrs>") {
    RC_SUBCASE([](UniformInitializerAttrs const &attrs) {
      RC_ASSERT(attrs.max_val >= attrs.min_val);
    });
  }
}

#include "op-attrs/regularizer_attrs.dtg.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Arbitrary<RegularizerAttrs>") {
    RC_SUBCASE([](RegularizerAttrs reg) {
      RC_ASSERT(reg.has<L1RegularizerAttrs>() || reg.has<L2RegularizerAttrs>());
    });
  }
}

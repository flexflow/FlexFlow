#include "compiler/machine_mapping/estimate_cost_across_split.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("estimate_cost_across_split") {
    SUBCASE("single edge across split") {
      SUBCASE("src and dst layers have same MachineView") {
        FAIL("TODO");
      }

      SUBCASE("src and dst layers have different MachineViews") {
        FAIL("TODO");
      }
    }

    SUBCASE("single tensor, multiple consumers across split") {
      SUBCASE("consumers have same view") {
        FAIL("TODO");
      }

      SUBCASE("consumers have non-overlapping views") {
        FAIL("TODO");
      }

      SUBCASE("consumers have different but overlapping views") {
        FAIL("TODO");
      }
    }

    SUBCASE("multiple tensors, multiple consumers across split") {
      FAIL("TODO");
    }
  }
}

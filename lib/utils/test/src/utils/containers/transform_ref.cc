#include "utils/containers/transform_ref.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform_ref") {
    int x = 5;

    transform_ref(x, [](int y) -> int { return y + 1; });

    CHECK(x == 6);
  }
}

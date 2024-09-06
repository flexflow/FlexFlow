#include "utils/rapidcheck/optional.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE(
      "Arbitrary<std::optional<T>> with T=", T, int, double, char) {
    RC_SUBCASE([](std::optional<T> o) {});
  }
}

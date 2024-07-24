#include "test/utils/doctest.h"
#include "test/utils/rapidcheck.h"
#include "utils/optional.h"
#include <rapidcheck.h>

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE(
      "Arbitrary<std::optional<T>> with T=", T, int, double, char) {
    RC_SUBCASE([](std::optional<T> o) {});
  }
}

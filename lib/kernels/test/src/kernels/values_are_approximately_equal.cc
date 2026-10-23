#include "kernels/values_are_approximately_equal.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("values_are_approximately_equal") {
    SUBCASE("values are the same") {
      bool result = values_are_approximately_equal(1.0f, 1.0f);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("values are within the error bound") {
      bool result = values_are_approximately_equal(1000.0f, 1001.0f, 0.01);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("values are very different") {
      bool result = values_are_approximately_equal(2.0f, 1.0f);
      bool correct = false;

      CHECK(result == correct);
    }
  }
}

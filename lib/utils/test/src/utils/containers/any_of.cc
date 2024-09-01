#include <doctest/doctest.h>
#include "utils/containers/any_of.h"
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("any_of(C, F)") {
    SUBCASE("has element matching condition") {
      std::vector<int> input = {1, 2, 3};

      bool result = any_of(input, [](int x) { return x > 1; });
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("does not have element matching condition") {
      std::vector<int> input = {1, 2, 3};

      bool result = any_of(input, [](int x) { return x > 5; });
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("input is empty") {
      std::vector<int> input = {};

      bool result = any_of(input, [](int x) { return true; });
      bool correct = false;

      CHECK(result == correct);
    }
  }
}

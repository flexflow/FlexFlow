#include "utils/containers/reversed.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reversed(std::vector<T>)") {
    SUBCASE("non-empty input") {
      std::vector<int> input = {1, 2, 3, 2};

      std::vector<int> result = reversed(input);
      std::vector<int> correct = {2, 3, 2, 1};

      CHECK(result == correct);
    }

    SUBCASE("empty input") {
      std::vector<int> input = {};

      std::vector<int> result = reversed(input);
      std::vector<int> correct = {};

      CHECK(result == correct);
    }
  }
}

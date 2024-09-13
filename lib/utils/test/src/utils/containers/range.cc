#include "utils/containers/range.h"
#include "utils/fmt/vector.h"
#include "utils/hash/unordered_set.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("range") {
    SUBCASE("step=1") {
      std::vector<int> result = range(0, 5);
      std::vector<int> correct = {0, 1, 2, 3, 4};
      CHECK(result == correct);
    }

    SUBCASE("step = 2") {
      std::vector<int> result = range(-2, 10, 2);
      std::vector<int> correct = {-2, 0, 2, 4, 6, 8};
      CHECK(result == correct);
    }

    SUBCASE("step = -1") {
      std::vector<int> result = range(5, 0, -1);
      std::vector<int> correct = {5, 4, 3, 2, 1};
      CHECK(result == correct);
    }

    SUBCASE("single argument") {
      std::vector<int> result = range(5);
      std::vector<int> correct = {0, 1, 2, 3, 4};
      CHECK(result == correct);
    }

    SUBCASE("start = end") {
      std::vector<int> result = range(5, 5);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("start > end") {
      std::vector<int> result = range(5, 4);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("start < end, step < 0") {
      std::vector<int> result = range(0, 10, -1);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }
  }
}

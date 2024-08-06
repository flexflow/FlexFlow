#include "utils/containers/range.h"
#include "utils/hash/unordered_set.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("range") {
    SUBCASE("basic integer range") {
      std::vector<int> result = range(0, 5);
      std::vector<int> correct = {0, 1, 2, 3, 4};
      CHECK(result == correct);
    }

    SUBCASE("integer range with step") {
      std::vector<int> result = range(-2, 10, 2);
      std::vector<int> correct = {-2, 0, 2, 4, 6, 8};
      CHECK(result == correct);
    }

    SUBCASE("negative integer range") {
      std::vector<int> result = range(5, 0, -1);
      std::vector<int> correct = {5, 4, 3, 2, 1};
      CHECK(result == correct);
    }

    SUBCASE("single argument range") {
      std::vector<int> result = range(5);
      std::vector<int> correct = {0, 1, 2, 3, 4};
      CHECK(result == correct);
    }

    SUBCASE("empty range") {
      std::vector<int> result = range(5, 5);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("empty range") {
      std::vector<int> result = range(5, 4);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }
  }
}

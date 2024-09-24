#include "utils/containers/vector_split.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>
#include <stdexcept>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Testing vector_split function") {
    std::vector<int> v = {1, 2, 3, 4, 5};

    SUBCASE("Normal case: idx = 2") {
      auto [prefix, postfix] = vector_split(v, 2);
      CHECK(prefix == std::vector<int>({1, 2}));
      CHECK(postfix == std::vector<int>({3, 4, 5}));
    }

    SUBCASE("Boundary case: idx = 0") {
      auto [prefix, postfix] = vector_split(v, 0);
      CHECK(prefix.empty());
      CHECK(postfix == std::vector<int>({1, 2, 3, 4, 5}));
    }

    SUBCASE("Boundary case: idx is the last index in the list") {
      auto [prefix, postfix] = vector_split(v, 5);
      CHECK(prefix == std::vector<int>({1, 2, 3, 4, 5}));
      CHECK(postfix.empty());
    }

    SUBCASE("Out of bounds case: idx = -1") {
      CHECK_THROWS_AS(vector_split(v, -1), std::out_of_range);
    }

    SUBCASE("Out of bounds case: idx == list_size + 1") {
      CHECK_THROWS_AS(vector_split(v, 6), std::out_of_range);
    }
  }
}

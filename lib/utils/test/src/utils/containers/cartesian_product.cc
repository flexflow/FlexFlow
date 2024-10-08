#include "utils/containers/cartesian_product.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cartesian_product") {

    SUBCASE("empty") {
      std::vector<std::unordered_set<int>> containers = {};
      std::unordered_set<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_set<std::vector<int>> correct = {{}};
      CHECK(result == correct);
    }

    SUBCASE("single container, one element") {
      std::vector<std::unordered_set<int>> containers = {{1}};
      std::unordered_set<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_set<std::vector<int>> correct = {{1}};
      CHECK(result == correct);
    }

    SUBCASE("single container, multiple elements") {
      std::vector<std::unordered_set<int>> containers = {{1, 2, 3}};
      std::unordered_set<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_set<std::vector<int>> correct = {{1}, {2}, {3}};
      CHECK(result == correct);
    }

    SUBCASE("multiple containers, one element each") {
      std::vector<std::unordered_set<int>> containers = {{1}, {2}, {3}};
      std::unordered_set<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_set<std::vector<int>> correct = {{1, 2, 3}};
      CHECK(result == correct);
    }

    SUBCASE("multiple containers, multiple elements") {
      std::vector<std::unordered_set<int>> containers = {{1, 2}, {3, 4}};
      std::unordered_set<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_set<std::vector<int>> correct = {
          {1, 3}, {1, 4}, {2, 3}, {2, 4}};
      CHECK(result == correct);
    }

    SUBCASE("1 empty container, 1 non-empty container") {
      std::vector<std::unordered_set<int>> containers = {{}, {2, 3}};
      std::unordered_set<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_set<std::vector<int>> correct = {};
      CHECK(result == correct);
    }
  }
}

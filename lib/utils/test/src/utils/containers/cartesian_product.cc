#include "utils/containers/cartesian_product.h"
#include "utils/fmt/unordered_multiset.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cartesian_product") {

    SUBCASE("empty") {
      std::vector<std::vector<int>> containers = {};
      std::unordered_multiset<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_multiset<std::vector<int>> correct = {{}};
      CHECK(result == correct);
    }

    SUBCASE("single container, one element") {
      std::vector<std::vector<int>> containers = {{1}};
      std::unordered_multiset<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_multiset<std::vector<int>> correct = {{1}};
      CHECK(result == correct);
    }

    SUBCASE("single container, multiple elements") {
      std::vector<std::vector<int>> containers = {{1, 2, 3}};
      std::unordered_multiset<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_multiset<std::vector<int>> correct = {{1}, {2}, {3}};
      CHECK(result == correct);
    }

    SUBCASE("multiple containers, one element each") {
      std::vector<std::vector<int>> containers = {{1}, {2}, {3}};
      std::unordered_multiset<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_multiset<std::vector<int>> correct = {{1, 2, 3}};
      CHECK(result == correct);
    }

    SUBCASE("multiple containers, multiple elements") {
      std::vector<std::vector<int>> containers = {{1, 2}, {3, 4}};
      std::unordered_multiset<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_multiset<std::vector<int>> correct = {
          {1, 3}, {1, 4}, {2, 3}, {2, 4}};
      CHECK(result == correct);
    }

    SUBCASE("multiple containers, duplicate elements") {
      std::vector<std::vector<int>> containers = {{1, 1}, {2, 3}};
      std::unordered_multiset<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_multiset<std::vector<int>> correct = {
          {1, 2}, {1, 3}, {1, 3}, {1, 2}};
      CHECK(result == correct);
    }

    SUBCASE("1 empty container, 1 non-empty container") {
      std::vector<std::vector<int>> containers = {{}, {2, 3}};
      std::unordered_multiset<std::vector<int>> result =
          cartesian_product(containers);
      std::unordered_multiset<std::vector<int>> correct = {};
      CHECK(result == correct);
    }
  }
}

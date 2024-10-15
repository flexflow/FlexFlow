#include "utils/containers/are_disjoint.h"
#include <doctest/doctest.h>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("are_disjoint") {
    SUBCASE("disjoint") {
      std::unordered_set<int> l = {1, 2, 3};
      std::unordered_set<int> r = {4, 5, 6};
      CHECK(are_disjoint(l, r));
    }
    SUBCASE("not disjoint") {
      std::unordered_set<int> l = {1, 2, 3, 4};
      std::unordered_set<int> r = {3, 4, 5, 6};
      CHECK_FALSE(are_disjoint(l, r));
    }

    SUBCASE("one empty set") {
      std::unordered_set<int> l = {1, 2};
      std::unordered_set<int> r = {};
      CHECK(are_disjoint(l, r));
    }
    SUBCASE("both empty sets") {
      std::unordered_set<int> l = {};
      std::unordered_set<int> r = {};
      CHECK(are_disjoint(l, r));
    }
  }
}

#include "utils/containers/get_one_of.h"
#include "utils/containers/contains.h"
#include <doctest/doctest.h>
#include <unordered_set>
using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_one_of") {
    SUBCASE("non-empty set") {
      std::unordered_set<int> s = {1, 2, 3};
      CHECK(contains(s, get_one_of(s)));
    }

    SUBCASE("empty set") {
      std::unordered_set<int> s = {};
      CHECK_THROWS(get_one_of(s));
    }
  }
}

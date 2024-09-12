#include "utils/containers/is_superseteq_of.h"
#include <doctest/doctest.h>
#include <unordered_set>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_superseteq_of") {
    std::unordered_set<int> super = {1, 2, 3, 4};

    SUBCASE("true containment") {
      std::unordered_set<int> sub = {1, 2, 3};
      CHECK(is_superseteq_of(super, sub));
    }

    SUBCASE("false containment") {
      std::unordered_set<int> sub = {1, 2, 5};
      CHECK_FALSE(is_superseteq_of(super, sub));
    }

    SUBCASE("reflexive") {
      CHECK(is_superseteq_of(super, super));
    }
  }
}

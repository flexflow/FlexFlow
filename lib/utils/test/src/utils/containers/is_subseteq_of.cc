#include "utils/containers/is_subseteq_of.h"
#include <doctest/doctest.h>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_subseteq_of") {
    std::unordered_set<int> s1 = {1, 2};
    std::unordered_set<int> s2 = {1, 2, 3};
    CHECK(is_subseteq_of(s1, s2) == true);
    CHECK(is_subseteq_of(s2, s1) == false);
    CHECK(is_subseteq_of(s1, s1) == true);
    CHECK(is_subseteq_of(s2, s2) == true);
  }
}

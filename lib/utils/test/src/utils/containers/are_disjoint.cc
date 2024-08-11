#include "utils/containers/are_disjoint.h"
#include <doctest/doctest.h>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("are_disjoint") {
    std::unordered_set<int> l = {1, 2, 3};
    std::unordered_set<int> r = {4, 5, 6};
    CHECK(are_disjoint(l, r));
    r.insert(3);
    CHECK_FALSE(are_disjoint(l, r));
  }
}

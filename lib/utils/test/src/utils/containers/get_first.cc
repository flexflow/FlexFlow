#include "utils/containers/get_first.h"
#include "utils/containers/contains.h"
#include <doctest/doctest.h>
#include <unordered_set>
using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_first") {
    std::unordered_set<int> s = {1, 2, 3};
    CHECK(contains(s, get_first(s)));
  }
}

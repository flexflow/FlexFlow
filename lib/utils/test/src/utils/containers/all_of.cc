#include "utils/containers/all_of.h"
#include <doctest/doctest.h>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("all_of") {
    std::vector<int> v = {2, 4, 6, 8};
    CHECK(all_of(v, [](int x) { return x % 2 == 0; }) == true);
    CHECK(all_of(v, [](int x) { return x % 4 == 0; }) == false);
  }
}

#include "utils/containers/count.h"
#include <doctest/doctest.h>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("count") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    CHECK(count(v, [](int x) { return x % 2 == 0; }) == 2);
    CHECK(count(v, [](int x) { return x % 2 == 1; }) == 3);
  }
}

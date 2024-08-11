#include "utils/containers/find.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    CHECK(find(v, 3) != v.cend());
    CHECK(find(v, 6) == v.cend());
  }
}

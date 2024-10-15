#include "utils/containers/contains.h"
#include <doctest/doctest.h>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("contains") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    CHECK(contains(v, 3));
    CHECK(!contains(v, 6));
  }
}

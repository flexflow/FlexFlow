#include "utils/containers/vector_split.h"
#include <doctest/doctest.h>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Testing vector_split function") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto result = vector_split(v, 2);
    std::vector<int> prefix = result.first;
    std::vector<int> postfix = result.second;
    CHECK(prefix == std::vector<int>({1, 2}));
    CHECK(postfix == std::vector<int>({3, 4, 5}));
  }
}

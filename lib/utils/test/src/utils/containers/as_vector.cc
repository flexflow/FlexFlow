#include "utils/containers/as_vector.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("as_vector") {
    std::unordered_set<int> s = {1, 2, 3};
    std::vector<int> result = as_vector(s);
    CHECK(result == std::vector<int>({3, 2, 1}));
  }
}

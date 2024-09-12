#include "utils/containers/as_vector.h"
#include "utils/containers/sorted.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("as_vector") {
    std::unordered_set<int> input = {1, 2, 3};
    std::vector<int> result = as_vector(input);
    std::vector<int> correct_sorted = {1, 2, 3};
    CHECK(sorted(result) == correct_sorted);
  }
}

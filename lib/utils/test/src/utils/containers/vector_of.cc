#include "utils/containers/vector_of.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <set>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("vector_of(std::set<T>)") {
    std::set<int> input = {2, 3, 1, 4};

    std::vector<int> result = vector_of(input);
    std::vector<int> correct = {1, 2, 3, 4};

    CHECK(result == correct);
  }
}

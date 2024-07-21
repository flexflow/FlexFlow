#include <doctest/doctest.h>
#include "utils/containers/get_only.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_only") {
    std::vector<int> input = {5};
    int result = get_only(input);
    int correct = 5;
    CHECK(result == correct);
  }

  TEST_CASE("get_only") {
    std::unordered_set<int> input = {5};
    int result = get_only(input);
    int correct = 5;
    CHECK(result == correct);
  }
}

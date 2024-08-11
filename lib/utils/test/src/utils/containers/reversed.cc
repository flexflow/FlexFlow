#include "utils/containers/reversed.h"
#include <doctest/doctest.h>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Testing the 'reversed' function") {
    std::vector<int> input_vec = {1, 2, 3, 4, 5};
    std::vector<int> result = reversed(input_vec);
    std::vector<int> expected = {5, 4, 3, 2, 1};

    // Checking the reversed sequence is as expected
    CHECK(result == expected);
  }
}

#include "utils/containers/extend_vector.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;
// checks rhs gets added to the end of lhs
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("extend_vector") {
    SUBCASE("non-empty vectors") {
      std::vector<int> lhs = {1, 2};
      std::vector<int> rhs = {3, 4};

      extend_vector(lhs, rhs);
      std::vector<int> correct = {1, 2, 3, 4};

      CHECK(lhs == correct);
    }
    // checks that lhs stays the same when the rhs is empty
    SUBCASE("empty rhs") {
      std::vector<int> lhs = {1, 2};
      std::vector<int> rhs = {};

      extend_vector(lhs, rhs);
      std::vector<int> correct = {1, 2};

      CHECK(lhs == correct);
    }
  }
}

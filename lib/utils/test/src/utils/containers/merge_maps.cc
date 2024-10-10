#include "utils/containers/merge_maps.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>
#include <unordered_map>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("merge_maps") {

    SUBCASE("disjoint keys") {
      std::unordered_map<int, std::string> lhs = {{1, "one"}, {2, "two"}};
      std::unordered_map<int, std::string> rhs = {{3, "three"}, {4, "four"}};

      std::unordered_map<int, std::string> result = merge_maps(lhs, rhs);
      std::unordered_map<int, std::string> correct = {
          {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};

      CHECK(result == correct);
    }

    SUBCASE("overlapping keys") {
      std::unordered_map<int, std::string> lhs = {{1, "one"}, {2, "two"}};
      std::unordered_map<int, std::string> rhs = {{2, "three"}, {3, "four"}};

      CHECK_THROWS(merge_maps(lhs, rhs));
    }
  }
}

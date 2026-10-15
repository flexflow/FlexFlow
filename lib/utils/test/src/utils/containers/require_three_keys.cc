#include "utils/containers/require_three_keys.h"
#include "test/utils/doctest/fmt/tuple.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("require_three_keys") {
    std::string s_a = "a";
    std::string s_b = "b";
    std::string s_c = "c";
    std::string s_d = "d";

    SUBCASE("input has more than three keys") {
      std::map<std::string, int> m = {
          {s_a, 1},
          {s_b, 5},
          {s_c, 2},
          {s_d, 6},
      };

      CHECK_THROWS(require_three_keys(m, s_b, s_a, s_c));
    }

    SUBCASE("input has less than three keys") {
      std::map<std::string, int> m = {
          {s_a, 1},
          {s_c, 2},
      };

      CHECK_THROWS(require_three_keys(m, s_b, s_a, s_c));
    }

    SUBCASE("input has a different three keys") {
      std::map<std::string, int> m = {
          {s_a, 1},
          {s_c, 2},
          {s_d, 6},
      };

      CHECK_THROWS(require_three_keys(m, s_b, s_a, s_c));
    }

    SUBCASE("input has exactly those three keys") {
      std::map<std::string, int> m = {
          {s_a, 1},
          {s_b, 5},
          {s_c, 2},
      };

      std::tuple<int, int, int> result = require_three_keys(m, s_b, s_a, s_c);
      std::tuple<int, int, int> correct = {5, 1, 2};

      CHECK(result == correct);
    }
  }
}

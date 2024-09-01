#include <doctest/doctest.h>
#include "op-attrs/dim_ordered/zip.h"
#include "op-attrs/ff_dim.dtg.h"
#include "utils/fmt/pair.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("zip(DimOrdered<Idx, T>, DimOrdered<Idx, T>)") {
    DimOrdered<ff_dim_t, int> lhs_input = {9, 9, 8, 9};
    DimOrdered<ff_dim_t, std::string> rhs_input = {"m", "m", "k", "l", "m"};

    SUBCASE("lhs is longer") {
      DimOrdered<ff_dim_t, std::pair<int, std::string>> result = zip(lhs_input, rhs_input);

      DimOrdered<ff_dim_t, std::pair<int, std::string>> correct = {
        {9, "m"},
        {9, "m"},
        {8, "k"},
        {9, "l"},
      };

      CHECK(result == correct);
    }

    SUBCASE("rhs is longer") {
      DimOrdered<ff_dim_t, std::pair<std::string, int>> result = zip(rhs_input, lhs_input);

      DimOrdered<ff_dim_t, std::pair<std::string, int>> correct = {
        {"m", 9},
        {"m", 9},
        {"k", 8},
        {"l", 9},
      };

      CHECK(result == correct);
    }
  }
}

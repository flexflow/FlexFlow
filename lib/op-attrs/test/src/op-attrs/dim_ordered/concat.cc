#include "op-attrs/dim_ordered/concat.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("concat(FFOrdered<T>, FFOrdered<T>)") {
    SUBCASE("inputs have elements") {
      FFOrdered<int> l_input = FFOrdered<int>{
        1, 3, 1
      };
      FFOrdered<int> r_input = FFOrdered<int>{
        2, 1
      };

      FFOrdered<int> result = concat(l_input, r_input);
      FFOrdered<int> correct = {
        1, 3, 1, 2, 1
      };

      CHECK(result == correct);
    }

    SUBCASE("inputs are empty") {
      FFOrdered<int> l_input = FFOrdered<int>{};
      FFOrdered<int> r_input = FFOrdered<int>{};

      FFOrdered<int> result = concat(l_input, r_input);
      FFOrdered<int> correct = {};

      CHECK(result == correct);
    }
  }

  TEST_CASE("concat(std::vector<FFOrdered<T>>)") {
    SUBCASE("inputs have elements") {
      std::vector<FFOrdered<int>> input = {
        {1},
        {2, 1},
        {1},
      };

      FFOrdered<int> result = concat(input);
      FFOrdered<int> correct = {
        1, 2, 1, 1,
      };

      CHECK(result == correct);
    }

    SUBCASE("no inputs") {
      std::vector<FFOrdered<int>> input = {};

      FFOrdered<int> result = concat(input);
      FFOrdered<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("inputs are empty") {
      std::vector<FFOrdered<int>> input = {
        {}, {}, {}
      };

      FFOrdered<int> result = concat(input);
      FFOrdered<int> correct = {};

      CHECK(result == correct);
    }
  }
}

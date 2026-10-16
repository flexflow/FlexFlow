#include "op-attrs/ff_ordered/ff_ordered_remove_suffix.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ff_ordered_remove_suffix") {
    SUBCASE("input ends with suffix") {
      FFOrdered<int> input = FFOrdered<int>{
          3,
          2,
          5,
          3,
          4,
      };

      FFOrdered<int> suffix = FFOrdered<int>{
          5,
          3,
          4,
      };

      FFOrdered<int> result = ff_ordered_remove_suffix(input, suffix);
      FFOrdered<int> correct = FFOrdered<int>{
          3,
          2,
      };

      CHECK(result == correct);
    }

    SUBCASE("input does not end with suffix") {
      FFOrdered<int> input = FFOrdered<int>{
          3,
          2,
          5,
          3,
          4,
      };

      FFOrdered<int> suffix = FFOrdered<int>{
          2,
          5,
          3,
      };

      CHECK_THROWS(ff_ordered_remove_suffix(input, suffix));
    }

    SUBCASE("suffix equals input") {
      FFOrdered<int> input = FFOrdered<int>{
          3,
          2,
          5,
          3,
          4,
      };

      FFOrdered<int> suffix = FFOrdered<int>{
          3,
          2,
          5,
          3,
          4,
      };

      FFOrdered<int> result = ff_ordered_remove_suffix(input, suffix);
      FFOrdered<int> correct = FFOrdered<int>{};

      CHECK(result == correct);
    }

    SUBCASE("suffix is longer than input") {
      FFOrdered<int> input = FFOrdered<int>{
          3,
          2,
          5,
          3,
          4,
      };

      FFOrdered<int> suffix = FFOrdered<int>{
          3,
          3,
          2,
          5,
          3,
          4,
      };

      CHECK_THROWS(ff_ordered_remove_suffix(input, suffix));
    }

    SUBCASE("suffix is empty") {
      FFOrdered<int> input = FFOrdered<int>{
          3,
          2,
          5,
          3,
          4,
      };

      FFOrdered<int> suffix = FFOrdered<int>{};

      FFOrdered<int> result = ff_ordered_remove_suffix(input, suffix);
      FFOrdered<int> correct = FFOrdered<int>{
          3,
          2,
          5,
          3,
          4,
      };

      CHECK(result == correct);
    }

    SUBCASE("suffix and input are empty") {
      FFOrdered<int> input = FFOrdered<int>{};
      FFOrdered<int> suffix = FFOrdered<int>{};

      FFOrdered<int> result = ff_ordered_remove_suffix(input, suffix);
      FFOrdered<int> correct = FFOrdered<int>{};

      CHECK(result == correct);
    }
  }
}

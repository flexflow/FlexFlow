#include "op-attrs/ff_ordered/ff_ordered_filter.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ff_ordered_filter") {
    FFOrdered<int> input = FFOrdered<int>{1, 2, 3, 4, 5};
    auto predicate = [](int x) { return x % 2 == 0; };

    FFOrdered<int> result = ff_ordered_filter(input, predicate);
    FFOrdered<int> correct = FFOrdered<int>{2, 4};
    CHECK(result == correct);
  }
}

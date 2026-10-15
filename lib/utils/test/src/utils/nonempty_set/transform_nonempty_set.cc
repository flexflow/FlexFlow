#include "utils/nonempty_set/transform_nonempty_set.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform_nonempty_set") {
    nonempty_set<int> input = nonempty_set<int>{
        -3,
        -2,
        1,
        2,
        4,
        6,
    };

    nonempty_set<nonnegative_int> result =
        transform_nonempty_set(input, [&](int x) -> nonnegative_int {
          return nonnegative_int{std::abs(x)};
        });

    nonempty_set<nonnegative_int> correct = nonempty_set<nonnegative_int>{
        1_n,
        2_n,
        3_n,
        4_n,
        6_n,
    };

    CHECK(result == correct);
  }
}

#include "op-attrs/dim_ordered/slice.h"
#include "test/utils/doctest.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE(
      "slice(DimOrdered<Idx, T>, std::optional<Idx>, std::optional<Idx>)") {
    FFOrdered<size_t> d = FFOrdered<size_t>{
        1,
        2,
        3,
        4,
    };

    FFOrdered<size_t> result = slice(d, std::nullopt, ff_dim_t{-1});
    FFOrdered<size_t> correct = FFOrdered<size_t>{
        1,
        2,
        3,
    };

    CHECK(result == correct);
  }
}

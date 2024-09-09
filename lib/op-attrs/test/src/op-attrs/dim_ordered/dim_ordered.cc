#include "doctest/doctest.h"
#include "op-attrs/dim_ordered/dim_ordered.h"
#include "test/utils/rapidcheck.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE_TEMPLATE(
      "Arbitrary<DimOrdered<int, T>> with T=", T, int, double, char) {
    RC_SUBCASE([](DimOrdered<int, T>) {});
  }

  TEST_CASE_TEMPLATE("Arbitrary<FFOrdered<T>> with T=", T, int, double, char) {
    RC_SUBCASE([](FFOrdered<T>) {});
  }
}

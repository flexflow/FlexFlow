#include "doctest/doctest.h"
#include "op-attrs/dim_ordered.h"
#include <rapidcheck.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE_TEMPLATE("RC", T, int, double, char) {
    CHECK(rc::check("generate",
                    [](FFOrdered<T> ff_dim, DimOrdered<int, T> dim) {}));
  }
}

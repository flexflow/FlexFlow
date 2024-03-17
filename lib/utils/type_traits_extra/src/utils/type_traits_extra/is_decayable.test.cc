#include "utils/type_traits_extra/is_decayable.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_decayable_v") {
    CHECK(is_decayable_v<int[]>);
    CHECK_FALSE(is_decayable_v<int *>);
    CHECK(is_decayable_v<int const>);
    CHECK_FALSE(is_decayable_v<int>);
  }

  TEST_CASE_TEMPLATE("is_decayable", T, int[], int *, int const, int) {
    CHECK(is_decayable<T>::value == is_decayable_v<T>);
  }
}

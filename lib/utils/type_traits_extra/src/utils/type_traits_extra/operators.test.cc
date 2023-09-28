#include "utils/testing.h"
#include "utils/type_traits_extra/operators.h"

struct no_operations_t { };

struct plusable {
  plusable operator+(plusable const &other) const;
};

struct minusable {
  minusable operator-(minusable const &other) const;
};

struct timesable {
  timesable operator*(timesable const &other) const;
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_plusable_v") {
    CHECK(is_plusable_v<plusable>);
    CHECK_FALSE(is_plusable_v<no_operations_t>);
  }

  TEST_CASE_TEMPLATE("is_plusable", T, plusable, no_operations_t) {
    CHECK(is_plusable<T>::value == is_plusable_v<T>);
  }

  TEST_CASE("is_minusable_v") {
    CHECK(is_minusable_v<minusable>);
    CHECK_FALSE(is_minusable_v<no_operations_t>);
  }

  TEST_CASE_TEMPLATE("is_minusable", T, minusable, no_operations_t) {
    CHECK(is_minusable<T>::value == is_minusable_v<T>);
  }

  TEST_CASE("is_timesable_v") {
    CHECK(is_timesable_v<timesable>);
    CHECK_FALSE(is_timesable_v<no_operations_t>);
  }

  TEST_CASE_TEMPLATE("is_timesable", T, timesable, no_operations_t) {
    CHECK(is_timesable<T>::value == is_timesable_v<T>);
  }
}

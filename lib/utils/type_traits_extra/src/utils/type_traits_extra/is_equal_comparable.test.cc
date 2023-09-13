#include "utils/testing.h"
#include "utils/type_traits_extra/is_equal_comparable.h"

struct equal_comparable {
  bool operator==(equal_comparable const &other) const;
};

struct not_equal_comparable {
  bool operator==(not_equal_comparable const &other) = delete;
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_equal_comparable_v") {
    CHECK(is_equal_comparable_v<equal_comparable>);
    CHECK_FALSE(is_equal_comparable_v<not_equal_comparable>);
  }

  TEST_CASE_TEMPLATE("is_equal_comparable", T, equal_comparable, not_equal_comparable) {
    CHECK(is_equal_comparable<T>::value == is_equal_comparable_v<T>);
  }
}

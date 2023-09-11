#include "utils/testing.h"
#include "utils/type_traits_extra/is_neq_comparable.h"

struct neq_comparable {
  bool operator!=(neq_comparable const &other) const;
};

struct not_neq_comparable {
  bool operator!=(neq_comparable const &other) = delete;
};

TEST_CASE("is_neq_comparable_v") {
  CHECK(is_neq_comparable_v<neq_comparable>);
  CHECK_FALSE(is_neq_comparable_v<not_neq_comparable>);
}

TEST_CASE_TEMPLATE("is_neq_comparable", T, neq_comparable, not_neq_comparable) {
  CHECK(is_neq_comparable<T>::value == is_neq_comparable_v<T>);
}

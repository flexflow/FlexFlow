#include "utils/testing.h"
#include "utils/type_traits_extra/is_static_castable.h"
#include <utility>

struct B {};

struct A {
  explicit operator B();
};

TEST_CASE("is_static_castable_v") {
  CHECK(is_static_castable_v<A, B>);
}

TEST_CASE_TEMPLATE("is_static_castable", T, std::pair<A, B>) {
  using From = typename T::first_type;
  using To = typename T::second_type;
  CHECK(is_static_castable_v<From, To> == is_static_castable<From, To>::value);
}

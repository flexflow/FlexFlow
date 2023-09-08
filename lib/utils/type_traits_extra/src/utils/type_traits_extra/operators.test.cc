#include "testing.h"
#include "utils/type_traits_extra/operators.h"

using namespace FlexFlow;
using namespace FlexFlow::test_types;

TEST_CASE("is_equal_comparable") {
  CHECK(is_equal_comparable_v<eq_t, eq_t>);
  CHECK(!is_equal_comparable_v<none_t, none_t>);
}

TEST_CASE("is_neq_comparable") {
  CHECK(is_neq_comparable_v<neq_t, neq_t>);
  CHECK(!is_neq_comparable_v<eq_t, eq_t>);
}

TEST_CASE("is_plusable") {
  CHECK(is_plusable_v<plusable_t, plusable_t>);
  CHECK(!is_plusable_v<cmp_t, cmp_t);
}

TEST_CASE("is_minusable") {
  CHECK(is_minusable_v<minusable_t, minusable_t>);
  CHECK(!is_minusable_v<plusable_t, plusable_t>);
}

TEST_CASE("is_timesable") {
  CHECK(is_timesable_v<timesable_t, timesable_t>);
  CHECK(!is_timesable_v<plusable_t, plusable_t>);
}

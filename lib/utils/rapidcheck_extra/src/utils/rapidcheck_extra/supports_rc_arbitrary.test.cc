#include "utils/testing.h"
#include "utils/rapidcheck_extra/supports_rc_arbitrary.h"

using namespace rc;

struct does_not_support_arbitrary {};

TEST_CASE("supports_rc_arbitrary_v") {
  CHECK(supports_rc_arbitrary_v<int>);
  CHECK_FALSE(supports_rc_arbitrary_v<does_not_support_arbitrary>);
}

TEST_CASE_TEMPLATE("supports_rc_arbitrary", T, int, does_not_support_arbitrary) {
  CHECK(supports_rc_arbitrary<T>::value == supports_rc_arbitrary_v<T>);
}

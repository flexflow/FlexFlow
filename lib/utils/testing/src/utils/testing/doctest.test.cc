#include "utils/testing.h"
#include "utils/testing/doctest.h"

TEST_CASE("CHECK_SAME_TYPE") {
  CHECK_SAME_TYPE(int, int);
}

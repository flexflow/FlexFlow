#include "utils/preprocessor_extra/stringize.h"
#include "utils/testing.h"

TEST_CASE("STRINGIZE") {
  CHECK(STRINGIZE(a, b, c) == "a, b, c");
  CHECK(STRINGIZE(a) == "a");
}

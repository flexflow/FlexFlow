#include "utils/testing.h"
#include "utils/fmt_extra/element_to_string.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("element_to_string") {
    CHECK(element_to_string(5) == "5");
    CHECK(element_to_string("hello \"your name\"") ==
          "\"hello \\\"your name\\\"\"");
    CHECK(element_to_string('a') == "'a'");
    CHECK(element_to_string('\'') == "'\\\''");
  }
}

#include "utils/testing.h"
#include "utils/fmt_extra/instances/tuple.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt instance for std::tuple") {
    CHECK(fmt::to_string(std::tuple{1, "hi", 5.3}) == "<1, \"hi\", 5.3>");
  }
}

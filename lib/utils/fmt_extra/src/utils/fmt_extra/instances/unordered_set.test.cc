#include "utils/fmt_extra/instances/unordered_set.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt instance for std::unordered_set") {
    CHECK(fmt::to_string(std::unordered_set<int>{2, 1, 4, 3}) ==
          "{1, 2, 3, 4}");
    CHECK(fmt::to_string(std::unordered_set<int>{}) == "{}");
  }
}

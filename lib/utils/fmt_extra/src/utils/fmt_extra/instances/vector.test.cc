#include "utils/fmt_extra/instances/vector.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt instance for std::vector") {
    CHECK(fmt::to_string(std::vector<int>{1, 2, 3, 4}) == "[1, 2, 3, 4]");
    CHECK(fmt::to_string(std::vector<int>{}) == "[]");
    CHECK(fmt::to_string(std::vector<std::string>{"aa", "bb", "cc"}) ==
          "[\"aa\", \"bb\", \"cc\"]");
  }
}

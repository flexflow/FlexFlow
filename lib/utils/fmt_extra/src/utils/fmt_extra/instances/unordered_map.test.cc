#include "utils/fmt_extra/instances/unordered_map.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt instance for std::unordered_map") {
    std::unordered_map<int, std::string> m = {
        {5, "yes"}, {1, "hello"}, {1000, ""}};
    CHECK(fmt::to_string(m) == "{1: \"hello\", 5: \"yes\", 1000: \"\"}");
  }
}

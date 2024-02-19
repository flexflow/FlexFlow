#include "utils/testing.h"
#include "utils/fmt_extra/instances/optional.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt instance for std::optional") {
    std::optional<std::string> x = "hello";
    std::optional<std::string> y = std::nullopt;
    CHECK(fmt::to_string(x) == x.value());
    CHECK(fmt::to_string(y) == "nullopt");
    CHECK(fmt::to_string(std::nullopt) == "nullopt");
  }
}

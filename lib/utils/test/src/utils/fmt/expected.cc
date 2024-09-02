#include "utils/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(tl::expected<int, std::string>)") {
    SUBCASE("expected") {
      tl::expected<int, std::string> input = 4;
      std::string result = fmt::to_string(input);
      std::string correct = "expected(4)";
      CHECK(result == correct);
    }

    SUBCASE("unexpected") {
      tl::expected<int, std::string> input = tl::unexpected("hello world");
      std::string result = fmt::to_string(input);
      std::string correct = "unexpected(hello world)";
      CHECK(result == correct);
    }
  }
}

#include "utils/optional-extra.h"
#include "utils/testing.h"
#include <exception>

struct custom_error_type : std::exception {};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unwrap") {
    SUBCASE("has value") {
      std::optional<int> x = 5;
      CHECK(unwrap(x, [] { throw custom_error_type{}; }) == 5);
    }

    SUBCASE("custom handler") {
      std::optional<int> x = std::nullopt;
      CHECK_THROWS_AS(unwrap(x, [] { throw custom_error_type{}; }),
                      custom_error_type);
    }

    SUBCASE("empty handler") {
      std::optional<int> x = std::nullopt;
      CHECK_THROWS_AS(unwrap(x, [] {}), std::bad_optional_access);
    }
  }
}

#include "testing.h"
#include "utils/optional-extra.h"
#include <exception>

struct custom_error_type : std::exception {};

TEST_CASE("unwrap") {
  std::optional<int> x = 5;
  CHECK(unwrap(x, [] { throw custom_error_type{}; }) == 5);
  CHECK_THROWS_AS(unwrap(x, [] { throw custom_error_type{}; }), custom_error_type);
  CHECK_THROWS_AS(unwrap(x, [] { }), std::bad_optional_access);
}

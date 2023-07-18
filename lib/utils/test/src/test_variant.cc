#include "doctest.h"
#include "utils/variant.h"

TEST_CASE("widen and narrow functions") {
  SUBCASE("widen function") {
    variant<int, float> v1 = 42;
    variant<int, float, double> result = widen<variant<int, float, double>>(v1);
    variant<int, float, double> expected = 42;
    CHECK(result == expected);
  }

  SUBCASE("narrow function") {
    variant<int, float, double> v2 = 3.14;
    optional<variant<int, float>> result = narrow<variant<int, float>>(v2);
    optional<variant<int, float>> expected = float(3.14);
    CHECK_FALSE(result.has_value()); // result should be empty due to narrowing
  }

  SUBCASE("cast function") {
    variant<int, float> v3 = 42;
    optional<variant<int, double>> result = cast<variant<int, double>>(v3);
    optional<variant<int, double>> expected = 42;
    CHECK(result == expected);
  }
}

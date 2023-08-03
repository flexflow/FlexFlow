#include "doctest.h"
#include "utils/variant.h"

TEST_CASE("widen and narrow functions") {
  SUBCASE("widen function") {
    variant<int, float> v1 = 42;
    variant<int, float, double> result = widen<variant<int, float, double>>(v1);
    variant<int, float, double> expected = 42;
    CHECK(result == expected);
  }

  SUBCASE("narrow function  fail") {
    variant<int, float, double> v2 = 3.14;//this is a doule, because 3.14 default to double
    optional<variant<int, float>> result = narrow<variant<int, float>>(v2);
    optional<variant<int, float>> expected = float(3.14);
    CHECK(!result.has_value()); // result should be empty due to narrowing
  }

  SUBCASE("narrow function  success") {
    variant<int, float, double> v2 = 3.14;//this is a doule, because 3.14 default to double
    optional<variant<int, double>> result = narrow<variant<int, double>>(v2);
    optional<variant<int, double>> expected = 3.14;
    CHECK(result == expected); // 
  }

  SUBCASE("cast function") {
    variant<int, float> v3 = 42;
    optional<variant<int, double>> result = cast<variant<int, double>>(v3);
    optional<variant<int, double>> expected = 42;
    CHECK(result == expected);
  }

}

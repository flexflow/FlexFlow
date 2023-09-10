#include "test/utils/doctest.h"
#include "utils/variant.h"

TEST_CASE("widen and narrow functions") {
  SUBCASE("widen function") {
    variant<int, float> v1 = 42;
    variant<int, float, double> result = widen<variant<int, float, double>>(v1);
    variant<int, float, double> expected = 42;
    CHECK(result == expected);
  }

  SUBCASE("narrow function  fail") {
    variant<int, float, double> v2 =
        3.14; // this is a doule, because 3.14 default to double
    optional<variant<int, float>> result = narrow<variant<int, float>>(v2);
    optional<variant<int, float>> expected = float(3.14);
    CHECK(!result.has_value()); // result should be empty due to narrowing
  }

  SUBCASE("narrow function  success") {
    variant<int, float, double> v2 =
        3.14; // this is a doule, because 3.14 default to double
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

TEST_CASE("Narrow and cast variants") {
  variant<int, float, double> original_variant = 42;

  // narrow
  optional<variant<int, double>> narrow_result =
      narrow<variant<int, double>>(original_variant);
  CHECK(narrow_result.has_value()); // assert narrow has value

  // cast
  optional<variant<int>> cast_result =
      cast<variant<int>>(narrow_result.value());
  CHECK(cast_result.has_value()); // assert cast has value
  CHECK(get<int>(cast_result.value()) == 42);
}

TEST_CASE("casting and widening a variant") {
  variant<int, float> smaller_variant = 42;
  variant<int, float, double> wider_variant;

  // Perform the cast operation
  optional<variant<int>> cast_result = cast<variant<int>>(smaller_variant);
  REQUIRE(cast_result); // Ensure the cast was successful

  // Perform the widening operation
  wider_variant = widen<variant<int, float, double>>(cast_result.value());

  // Check the result
  CHECK(get<int>(wider_variant) == 42);
}

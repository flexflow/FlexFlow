#include "utils/variant.h"
#include "test/utils/doctest/fmt/optional.h"
#include "test/utils/doctest/fmt/variant.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("widen and narrow functions") {
    SUBCASE("widen function") {
      std::variant<int, float> v1 = 42;
      std::variant<int, float, double> result =
          widen<std::variant<int, float, double>>(v1);
      std::variant<int, float, double> expected = 42;
      CHECK(result == expected);
    }

    SUBCASE("narrow function  fail") {
      std::variant<int, float, double> v2 =
          3.14; // this is a doule, because 3.14 default to double
      std::optional<std::variant<int, float>> result =
          narrow<std::variant<int, float>>(v2);
      std::optional<std::variant<int, float>> expected = float(3.14);
      CHECK(!result.has_value()); // result should be empty due to narrowing
    }

    SUBCASE("narrow function  success") {
      std::variant<int, float, double> v2 =
          3.14; // this is a doule, because 3.14 default to double
      std::optional<std::variant<int, double>> result =
          narrow<std::variant<int, double>>(v2);
      std::optional<std::variant<int, double>> expected = 3.14;
      CHECK(result == expected); //
    }

    SUBCASE("cast function") {
      std::variant<int, float> v3 = 42;
      std::optional<std::variant<int, double>> result =
          cast<std::variant<int, double>>(v3);
      std::optional<std::variant<int, double>> expected = 42;
      CHECK(result == expected);
    }
  }

  TEST_CASE("Narrow and cast variants") {
    std::variant<int, float, double> original_variant = 42;

    // narrow
    std::optional<std::variant<int, double>> narrow_result =
        narrow<std::variant<int, double>>(original_variant);
    CHECK(narrow_result.has_value()); // assert narrow has value

    // cast
    std::optional<std::variant<int>> cast_result =
        cast<std::variant<int>>(narrow_result.value());
    CHECK(cast_result.has_value()); // assert cast has value
    CHECK(get<int>(cast_result.value()) == 42);
  }

  TEST_CASE("casting and widening a variant") {
    std::variant<int, float> smaller_variant = 42;
    std::variant<int, float, double> wider_variant;

    // Perform the cast operation
    std::optional<std::variant<int>> cast_result =
        cast<std::variant<int>>(smaller_variant);
    REQUIRE(cast_result); // Ensure the cast was successful

    // Perform the widening operation
    wider_variant =
        widen<std::variant<int, float, double>>(cast_result.value());

    // Check the result
    CHECK(get<int>(wider_variant) == 42);
  }

  TEST_CASE("Arbitrary<std::variant>") {
    RC_SUBCASE("valid type", [](std::variant<int, float> v) {
      return std::holds_alternative<int>(v) || std::holds_alternative<float>(v);
    });
  }
}

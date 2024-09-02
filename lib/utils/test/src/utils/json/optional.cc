#include <doctest/doctest.h>
#include "utils/json/optional.h"
#include "test/utils/doctest/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("adl_serializer<std::optional<T>>") {
    SUBCASE("to_json") {
      SUBCASE("has value") {
        std::optional<int> input = 5;

        nlohmann::json result = input;
        nlohmann::json correct = 5;

        CHECK(result == correct);
      }

      SUBCASE("has nullopt") {
        std::optional<int> input = std::nullopt;

        nlohmann::json result = input;
        nlohmann::json correct = nullptr;

        CHECK(result == correct);
      }
    }

    SUBCASE("from_json") {
      SUBCASE("has value") {
        nlohmann::json input = 5;
        
        std::optional<int> result = input;
        std::optional<int> correct = 5;
        
        CHECK(result == correct);
      }

      SUBCASE("has nullopt") {
        nlohmann::json input = nullptr;

        std::optional<int> result = input.get<std::optional<int>>();
        std::optional<int> correct = std::nullopt;

        CHECK(result == correct);
      }
    }
  }
}

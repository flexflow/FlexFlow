#include "utils/containers/optional_all_of.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>
#include <optional>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("optional_all_of") {
    std::vector<int> input = {2, 4, 6, 8};

    SUBCASE("All values satisfy condition") {
      auto f = [](int x) -> std::optional<bool> { return x % 2 == 0; };
      std::optional<bool> correct = true;
      std::optional<bool> result = optional_all_of(input, f);
      CHECK(correct == result);
    }

    SUBCASE("One value returns nullopt") {
      auto f = [](int x) -> std::optional<bool> {
        if (x == 6) {
          return std::nullopt;
        }
        return x % 2 == 0;
      };
      std::optional<bool> correct = std::nullopt;
      std::optional<bool> result = optional_all_of(input, f);
      CHECK(correct == result);
    }

    SUBCASE("Empty container") {
      auto f = [](int x) -> std::optional<bool> { return true; };
      std::optional<bool> correct = true;
      std::optional<bool> result = optional_all_of(input, f);
      CHECK(correct == result);
    }
  }
}

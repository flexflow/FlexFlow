#include "utils/one_to_many/one_to_many_transform_values.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("one_to_many_transform_values") {
    OneToMany<std::string, int> input = OneToMany<std::string, int>{
        {
            "a",
            {1, 2, 3},
        },
        {
            "b",
            {4, 6},
        },
    };

    SUBCASE("does not produce duplicate values") {
      auto func = [](int x) -> std::string { return fmt::to_string(x); };

      OneToMany<std::string, std::string> result =
          one_to_many_transform_values(input, func);

      OneToMany<std::string, std::string> correct = {
          {
              "a",
              {"1", "2", "3"},
          },
          {
              "b",
              {"4", "6"},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does produce duplicate values within a value set") {
      auto func = [](int x) -> std::string {
        if (x < 4) {
          return "a";
        } else {
          return "b";
        }
      };

      OneToMany<std::string, std::string> result =
          one_to_many_transform_values(input, func);

      OneToMany<std::string, std::string> correct = {
          {
              "a",
              {"a"},
          },
          {
              "b",
              {"b"},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does produce duplicate values across value sets") {
      auto func = [](int x) -> std::string { return ""; };

      CHECK_THROWS(one_to_many_transform_values(input, func));
    }
  }
}

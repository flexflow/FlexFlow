#include "utils/many_to_one/many_to_one_transform_values.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("many_to_one_transform_values") {
    ManyToOne<std::string, int> input = ManyToOne<std::string, int>{
        {
            {"one", "ONE"},
            1,
        },
        {
            {"two", "TWO", "tWo"},
            2,
        },
    };

    SUBCASE("does not produce duplicate values") {
      auto func = [](int x) -> std::string { return fmt::to_string(x); };

      ManyToOne<std::string, std::string> result =
          many_to_one_transform_values(input, func);

      ManyToOne<std::string, std::string> correct = {
          {
              {"one", "ONE"},
              "1",
          },
          {
              {"two", "TWO", "tWo"},
              "2",
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does produce duplicate values") {
      auto func = [](int x) -> std::string { return ""; };

      ManyToOne<std::string, std::string> result =
          many_to_one_transform_values(input, func);

      ManyToOne<std::string, std::string> correct = {
          {
              {"one", "ONE", "two", "TWO", "tWo"},
              "",
          },
      };

      CHECK(result == correct);
    }
  }
}

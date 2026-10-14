#include "utils/one_to_many/one_to_many_transform_keys.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("one_to_many_transform_keys") {
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

    SUBCASE("does not produce duplicate keys") {
      auto func = [](std::string const &x) -> int { return x.at(0); };

      OneToMany<int, int> result = one_to_many_transform_keys(input, func);

      OneToMany<int, int> correct = {
          {
              0x61,
              {1, 2, 3},
          },
          {
              0x62,
              {4, 6},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does produce duplicate keys") {
      auto func = [](std::string const &x) -> int { return x.size(); };

      OneToMany<int, int> result = one_to_many_transform_keys(input, func);

      OneToMany<int, int> correct = {
          {
              1,
              {1, 2, 3, 4, 6},
          },
      };

      CHECK(result == correct);
    }
  }
}

#include "utils/many_to_one/many_to_one_transform_keys.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("many_to_one_transform_keys") {
    ManyToOne<std::string, int> input = ManyToOne<std::string, int>{
        {
            {"one", "ONE"},
            1,
        },
        {
            {"two", "TWO", "dwo"},
            2,
        },
    };

    SUBCASE("does not produce duplicate keys") {
      auto func = [](std::string const &x) -> int { return x.at(0); };

      ManyToOne<int, int> result = many_to_one_transform_keys(input, func);

      ManyToOne<int, int> correct = {
          {
              {0x6f, 0x4f},
              1,
          },
          {
              {0x74, 0x54, 0x64},
              2,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does produce duplicate keys within a key set") {
      auto func = [](std::string const &x) -> int { return x.at(1); };

      ManyToOne<int, int> result = many_to_one_transform_keys(input, func);

      ManyToOne<int, int> correct = {
          {
              {0x6e, 0x4e},
              1,
          },
          {
              {0x77, 0x57},
              2,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does produce duplicate keys across key sets") {
      auto func = [](std::string const &x) -> int { return 0; };

      CHECK_THROWS(many_to_one_transform_keys(input, func));
    }
  }
}

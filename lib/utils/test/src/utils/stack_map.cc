#include "utils/stack_map.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("stack_map") {
    stack_map<int, int, 5> map;

    SUBCASE("operator[]") {
      map[1] = 10;
      map[2] = 20;

      CHECK(map[1] == 10);
      CHECK(map[2] == 20);
    }

    SUBCASE("insert") {
      map.insert(1, 10);
      map.insert(2, 20);

      CHECK(map[1] == 10);
      CHECK(map[2] == 20);
    }

    SUBCASE("at") {
      map[1] = 10;
      map[2] = 20;

      CHECK(map.at(1) == 10);
      CHECK(map.at(2) == 20);
      CHECK(map.at(1) != 20);

      stack_map<int, int, 5> const &const_map = map;
      CHECK(const_map.at(1) == 10);
      CHECK(const_map.at(2) == 20);
    }

    SUBCASE("Iterator") {
      map[1] = 10;
      map[2] = 20;
      map[3] = 30;

      std::vector<std::pair<int, int>> expected = {{1, 10}, {2, 20}, {3, 30}};
      std::vector<std::pair<int, int>> actual = map;
      CHECK(actual == expected);
    }
  }
}

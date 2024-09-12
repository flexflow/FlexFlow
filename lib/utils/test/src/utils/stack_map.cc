#include "utils/stack_map.h"
#include "test/utils/doctest.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("stack_map") {
    stack_map<int, int, 5> map;
    // Test the [] operator to insert and access elements
    SUBCASE("BracketOperator") {
      map[1] = 10;
      map[2] = 20;

      CHECK(map[1] == 10);
      CHECK(map[2] == 20);
    }

    // Test the insert() function
    SUBCASE("Insert") {
      map.insert(1, 10);
      map.insert(2, 20);

      CHECK(map[1] == 10);
      CHECK(map[2] == 20);
    }

    // Test the at() function to access elements
    SUBCASE("At") {
      map[1] = 10;
      map[2] = 20;

      CHECK(map.at(1) == 10);
      CHECK(map.at(2) == 20);
      CHECK(map.at(1) != 20);
      // Test const version of at() function
      stack_map<int, int, 5> const &const_map = map;
      CHECK(const_map.at(1) == 10);
      CHECK(const_map.at(2) == 20);
    }

    // Test the begin() and end() functions for iterator
    SUBCASE("Iterator") {
      map[1] = 10;
      map[2] = 20;
      map[3] = 30;

      std::vector<std::pair<int, int>> expected = {{1, 10}, {2, 20}, {3, 30}};
      std::vector<std::pair<int, int>> actual = map;
      CHECK_WITHOUT_STRINGIFY(actual == expected);
    }
  }
}

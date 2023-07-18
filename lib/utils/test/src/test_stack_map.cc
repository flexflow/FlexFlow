#include "doctest.h"
#include "utils/stack_map.h"

using namespace FlexFlow;

// Test fixture for stack_map tests
struct StackMapTestFixture {
  stack_map<int, int, 5> map;
};

// Test the [] operator to insert and access elements
TEST_CASE_FIXTURE(StackMapTestFixture, "BracketOperator") {
  map[1] = 10;
  map[2] = 20;

  CHECK_EQ(map[1], 10);
  CHECK_EQ(map[2], 20);
}

// Test the insert() function
TEST_CASE_FIXTURE(StackMapTestFixture, "Insert") {
  map.insert(1, 10);
  map.insert(2, 20);

  CHECK_EQ(map[1], 10);
  CHECK_EQ(map[2], 20);
}

// Test the at() function to access elements
TEST_CASE_FIXTURE(StackMapTestFixture, "At") {
  map[1] = 10;
  map[2] = 20;

  CHECK_EQ(map.at(1), 10);
  CHECK_EQ(map.at(2), 20);

  // Test const version of at() function
  stack_map<int, int, 5> const &const_map = map;
  CHECK_EQ(const_map.at(1), 10);
  CHECK_EQ(const_map.at(2), 20);
}

// Test the begin() and end() functions for iterator
TEST_CASE_FIXTURE(StackMapTestFixture, "Iterator") {
  map[1] = 10;
  map[2] = 20;
  map[3] = 30;

  std::vector<std::pair<int, int>> expected = {{1, 10}, {2, 20}, {3, 30}};

  int index = 0;
  for (auto it = map.begin(); it != map.end(); ++it) {
    CHECK_EQ(*it, expected[index++]);
  }

  // Test const version of iterators
  stack_map<int, int, 5> const &const_map = map;
  index = 0;
  for (auto it = const_map.begin(); it != const_map.end(); ++it) {
    CHECK_EQ(*it, expected[index++]);
  }
}

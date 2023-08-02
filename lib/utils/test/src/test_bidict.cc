#include "doctest.h"
#include "utils/bidict.h"

using namespace FlexFlow;

TEST_CASE("bidict"){

  bidict<int, std::string> dict;
// Test the equate() function
  SUBCASE("Equate") {
  dict.equate(1, "one");
  dict.equate(2, "two");

  CHECK_EQ(dict.at_l(1), "one");
  CHECK_EQ(dict.at_r("one"), 1);
  CHECK_EQ(dict.at_l(2), "two");
  CHECK_EQ(dict.at_r("two"), 2);
}

// Test the erase_l() function
SUBCASE("EraseL") {
  dict.equate(1, "one");
  dict.equate(2, "two");

  dict.erase_l(1);

  CHECK_EQ(dict.size(), 1);
  CHECK_THROWS_AS(dict.at_l(1), std::out_of_range);
  CHECK_EQ(dict.at_r("two"), 2);
}

// Test the erase_r() function
SUBCASE("EraseR") {
  dict.equate(1, "one");
  dict.equate(2, "two");

  dict.erase_r("one");

  CHECK_EQ(dict.size(), 1);
  CHECK_THROWS_AS(dict.at_r("one"), std::out_of_range);
  CHECK_EQ(dict.at_l(2), "two");
}

// Test the reversed() function
SUBCASE("Reversed") {
  dict.equate(1, "one");
  dict.equate(2, "two");

  bidict<std::string, int> reversed_dict = dict.reversed();

  CHECK_EQ(reversed_dict.at_l("one"), 1);
  CHECK_EQ(reversed_dict.at_r(2), "two");
}

// Test the size() function
SUBCASE("Size") {
  dict.equate(1, "one");
  dict.equate(2, "two");

  CHECK_EQ(dict.size(), 2);
}

}
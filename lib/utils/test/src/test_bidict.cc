#include "test/utils/doctest.h"
#include "utils/bidict.h"

using namespace FlexFlow;

TEST_CASE("bidict") {
  bidict<int, std::string> dict;
  dict.equate(1, "one");
  dict.equate(2, "two");

  // Test the equate() function
  SUBCASE("Equate") {
    CHECK(dict.at_l(1) == "one");
    CHECK(dict.at_r("one") == 1);
    CHECK(dict.at_l(2) == "two");
    CHECK(dict.at_r("two") == 2);
  }

  // Test the erase_l() function
  SUBCASE("EraseL") {
    dict.erase_l(1);
    CHECK(dict.size() == 1);
    CHECK_THROWS_AS(dict.at_l(1), std::out_of_range);
    CHECK(dict.at_r("two") == 2);
  }

  // Test the erase_r() function
  SUBCASE("EraseR") {
    dict.erase_r("one");
    CHECK(dict.size() == 1);
    CHECK_THROWS_AS(dict.at_r("one"), std::out_of_range);
    CHECK(dict.at_l(2) == "two");
  }

  // Test the reversed() function
  SUBCASE("Reversed") {
    bidict<std::string, int> reversed_dict = dict.reversed();
    CHECK(reversed_dict.at_l("one") == 1);
    CHECK(reversed_dict.at_r(2) == "two");
  }

  // Test the size() function
  SUBCASE("Size") {
    CHECK(dict.size() == 2);
  }

  SUBCASE("implicitly convert to std::unordered_map") {
    std::unordered_map<int, std::string> res = dict;
    std::unordered_map<int, std::string> expected = {{1, "one"}, {2, "two"}};
    CHECK(res == expected);
  }

  SUBCASE("begin") {
    auto it = dict.begin();
    CHECK(it->first == 2);
    CHECK(it->second == "two");
  }

  SUBCASE("end") {
    auto it = dict.end();
    CHECK(it == dict.end());
  }
}

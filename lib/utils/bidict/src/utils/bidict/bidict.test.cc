#include "utils/bidict/bidict.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("bidict::bidict()") {
    bidict<int, int> x;
    CHECK(x.size() == 0);
    CHECK(x.empty());
  }

  TEST_CASE("bidict equality operators") {
    bidict<int, std::string> lhs;
    lhs.equate(1, "one");
    lhs.equate(2, "two");

    bidict<int, std::string> rhs;
    rhs.equate(1, "one");
    rhs.equate(2, "two");

    bidict<int, std::string> rhs_different_value;
    rhs_different_value.equate(1, "one");
    rhs_different_value.equate(2, "TWO");

    bidict<int, std::string> rhs_different_key;
    rhs_different_key.equate(1, "one");
    rhs_different_key.equate(3, "two");

    bidict<int, std::string> rhs_smaller;
    rhs_different_key.equate(1, "one");

    SUBCASE("operator==(bidict<L, R> const &, bidict<L, R> const &)") {
      CHECK(lhs == lhs);
      CHECK(rhs == rhs);

      CHECK(lhs == rhs);
      CHECK(!(lhs == rhs_different_key));
      CHECK(!(lhs == rhs_different_value));
      CHECK(!(lhs == rhs_smaller));
    }

    SUBCASE("operator!=(bidict<L, R> const &, bidict<L, R> const &)") {
      CHECK(!(lhs != lhs));
      CHECK(!(rhs != rhs));

      CHECK(!(lhs != rhs));
      CHECK(lhs != rhs_different_key);
      CHECK(lhs != rhs_different_value);
      CHECK(lhs != rhs_smaller);
    }
  }

  TEST_CASE("bidict::") {
    bidict<int, std::string> dict;
    dict.equate(1, "one");
    dict.equate(2, "two");

    SUBCASE("at_l(L const &)") {
      CHECK(dict.at_l(1) == "one");
      CHECK(dict.at_l(2) == "two");
    }

    SUBCASE("at_r(R const &)") {
      CHECK(dict.at_r("one") == 1);
      CHECK(dict.at_r("two") == 2);
    }

    SUBCASE("erase_l(L const &)") {
      dict.erase_l(1);
      CHECK(dict.size() == 1);
      CHECK_THROWS_AS(dict.at_l(1), std::out_of_range);
      CHECK(dict.at_r("two") == 2);
    }

    SUBCASE("erase_r(R const &)") {
      dict.erase_r("one");
      CHECK(dict.size() == 1);
      CHECK_THROWS_AS(dict.at_r("one"), std::out_of_range);
      CHECK(dict.at_l(2) == "two");
    }

    SUBCASE("reversed()") {
      bidict<std::string, int> reversed_dict = dict.reversed();
      CHECK(reversed_dict.at_l("one") == 1);
      CHECK(reversed_dict.at_r(2) == "two");
    }

    SUBCASE("size()") {
      CHECK(dict.size() == 2);
    }

    SUBCASE("implicitly convert to std::unordered_map") {
      std::unordered_map<int, std::string> res = dict;
      std::unordered_map<int, std::string> expected = {{1, "one"}, {2, "two"}};
      CHECK(res == expected);
    }

    SUBCASE("begin()") {
      auto it = dict.begin();
      CHECK(it->first == 2);
      CHECK(it->second == "two");
    }

    SUBCASE("end()") {
      auto it = dict.end();
      CHECK(it == dict.end());
    }

    SUBCASE("find_l(L const &)") {
      // for valid keys
      auto it = dict.find_l(1);
      CHECK(it != dict.end());
      CHECK(it->first == 1);
      CHECK(it->second == "one");

      // for missing keys
      CHECK(dict.find_l(3) == dict.end());
    }

    SUBCASE("find_r(R const &)") {
      // for valid keys
      auto it = dict.find_r("one");
      CHECK(it != dict.end());
      CHECK(it->first == 1);
      CHECK(it->second == "one");

      // for missing keys
      CHECK(dict.find_r("three") == dict.end());
    }
  }
}

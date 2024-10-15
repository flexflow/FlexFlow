#include "utils/containers/is_submapeq_of.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_map>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_submapeq_of") {
    std::unordered_map<int, std::string> super = {
        {1, "one"}, {2, "two"}, {3, "three"}};

    SUBCASE("keys and values match") {
      std::unordered_map<int, std::string> sub = {{1, "one"}, {2, "two"}};
      CHECK(is_submapeq_of(sub, super));
    }

    SUBCASE("keys and values don't match") {
      std::unordered_map<int, std::string> sub = {{1, "one"}, {4, "four"}};
      CHECK_FALSE(is_submapeq_of(sub, super));
    }

    SUBCASE("keys match but values don't") {
      std::unordered_map<int, std::string> sub = {{1, "wrong_value"},
                                                  {2, "two"}};
      CHECK_FALSE(is_submapeq_of(sub, super));
    }

    SUBCASE("values match but keys don't") {
      std::unordered_map<int, std::string> sub = {{5, "one"}, {6, "two"}};
      CHECK_FALSE(is_submapeq_of(sub, super));
    }

    SUBCASE("sub is a superset of super") {
      std::unordered_map<int, std::string> sub = {
          {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};
      CHECK_FALSE(is_submapeq_of(sub, super));
    }
  }
}

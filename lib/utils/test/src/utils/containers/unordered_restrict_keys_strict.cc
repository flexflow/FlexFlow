#include "utils/containers/unordered_restrict_keys_strict.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_restrict_keys_strict") {
    std::unordered_map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}};

    SUBCASE("requested keys are a proper subset") {
      std::unordered_set<int> mask = {2, 3};
      std::unordered_map<int, std::string> result =
          unordered_restrict_keys_strict(m, mask);
      std::unordered_map<int, std::string> correct = {{2, "two"}, {3, "three"}};
      CHECK(result == correct);
    }

    SUBCASE("requested keys are the full set") {
      std::unordered_set<int> mask = {1, 2, 3};
      std::unordered_map<int, std::string> result =
          unordered_restrict_keys_strict(m, mask);
      std::unordered_map<int, std::string> correct = m;
      CHECK(result == correct);
    }

    SUBCASE("requested keys are a superset") {
      std::unordered_set<int> mask = {1, 2, 3, 4};
      CHECK_THROWS(unordered_restrict_keys_strict(m, mask));
    }

    SUBCASE("requested keys overlap") {
      std::unordered_set<int> mask = {2, 3, 4};
      CHECK_THROWS(unordered_restrict_keys_strict(m, mask));
    }

    SUBCASE("requested keys are disjoint") {
      std::unordered_set<int> mask = {5, 6};
      CHECK_THROWS(unordered_restrict_keys_strict(m, mask));
    }
  }
}

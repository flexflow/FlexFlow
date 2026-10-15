#include "utils/containers/restrict_keys_strict.h"
#include "test/utils/doctest/fmt/map.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("restrict_keys_strict") {
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}, {3, "three"}};

    SUBCASE("requested keys are a proper subset") {
      std::set<int> mask = {2, 3};
      std::map<int, std::string> result = restrict_keys_strict(m, mask);
      std::map<int, std::string> correct = {{2, "two"}, {3, "three"}};
      CHECK(result == correct);
    }

    SUBCASE("requested keys are the full set") {
      std::set<int> mask = {1, 2, 3};
      std::map<int, std::string> result = restrict_keys_strict(m, mask);
      std::map<int, std::string> correct = m;
      CHECK(result == correct);
    }

    SUBCASE("requested keys are a superset") {
      std::set<int> mask = {1, 2, 3, 4};
      CHECK_THROWS(restrict_keys_strict(m, mask));
    }

    SUBCASE("requested keys overlap") {
      std::set<int> mask = {2, 3, 4};
      CHECK_THROWS(restrict_keys_strict(m, mask));
    }

    SUBCASE("requested keys are disjoint") {
      std::set<int> mask = {5, 6};
      CHECK_THROWS(restrict_keys_strict(m, mask));
    }
  }
}

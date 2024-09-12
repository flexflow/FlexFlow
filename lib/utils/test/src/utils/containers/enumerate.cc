#include "utils/containers/enumerate.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/keys.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/containers/values.h"
#include "utils/fmt/map.h"
#include "utils/fmt/pair.h"
#include "utils/fmt/unordered_multiset.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_set>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("enumerate(std::vector<T>)") {
    std::vector<std::string> input = {"zero", "one", "two", "three"};

    std::map<int, std::string> correct = {
        {0, "zero"},
        {1, "one"},
        {2, "two"},
        {3, "three"},
    };

    std::map<int, std::string> result = enumerate(input);

    CHECK(result == correct);

    SUBCASE("check iteration order") {
      std::vector<std::pair<int const, std::string>> iterated_result =
          as_vector(result);
      std::vector<std::pair<int const, std::string>> correct_iteration_order = {
          {0, "zero"},
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };

      CHECK(iterated_result == correct_iteration_order);
    }
  }

  TEST_CASE("enumerate(std::unordered_set<T>)") {
    std::unordered_set<std::string> input = {"zero", "one", "two", "three"};

    std::unordered_set<int> correct_keys = {0, 1, 2, 3};
    std::unordered_multiset<std::string> correct_values = {
        "zero", "one", "two", "three"};
    std::map<int, std::string> result = enumerate(input);

    CHECK(keys(result) == correct_keys);
    CHECK(unordered_multiset_of(values(result)) == correct_values);
  }
}

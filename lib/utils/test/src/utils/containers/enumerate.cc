#include "utils/containers/enumerate.h"
#include "test/utils/doctest/fmt/map.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/doctest/fmt/pair.h"
#include "utils/containers/keys.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include <doctest/doctest.h>
#include <string>

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
          vector_of(result);
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

    std::map<int, std::string> correct = {
        {0, "zero"},
        {1, "one"},
        {2, "two"},
        {3, "three"},
    };

    std::map<int, std::string> result = enumerate(input);

    std::unordered_set<int> result_keys = keys(correct);
    std::unordered_set<std::string> result_values =
        unordered_set_of(values(correct));

    std::unordered_set<int> correct_keys = {0, 1, 2, 3};
    std::unordered_set<std::string> correct_values = input;

    CHECK(result_keys == correct_keys);
    CHECK(result_values == correct_values);
  }
}

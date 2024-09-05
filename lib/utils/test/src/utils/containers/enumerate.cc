#include "utils/containers/enumerate.h"
#include "utils/containers/as_vector.h"
#include "utils/fmt/map.h"
#include "utils/fmt/pair.h"
#include "utils/fmt/vector.h"
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

    std::map<int, std::string> correct = {
        {0, "zero"},
        {1, "one"},
        {2, "two"},
        {3, "three"},
    };
  }
}

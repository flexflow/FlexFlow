#include "utils/containers/are_all_same.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("are_all_same(std::vector<T>)") {
    SUBCASE("input is empty") {
      std::vector<int> input = {};

      bool result = are_all_same(input);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("input elements are all same") {
      std::vector<int> input = {1, 1, 1};

      bool result = are_all_same(input);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("input elements are not all same") {
      std::vector<int> input = {1, 1, 2, 1};

      bool result = are_all_same(input);
      bool correct = false;

      CHECK(result == correct);
    }
  }
}

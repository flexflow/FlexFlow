#include "utils/containers/flatmap.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flatmap") {
    SUBCASE("same data-type") {
      auto get_factors = [](int x) -> std::vector<int> {
        // Returns a vector of factors of x
        std::vector<int> factors;
        for (int i = 1; i <= x; i++) {
          if (x % i == 0) {
            factors.push_back(i);
          }
        }
        return factors;
      };

      std::vector<int> input = {2, 3, 4, 5};
      std::vector<int> result = flatmap(input, get_factors);
      std::vector<int> correct = {1, 2, 1, 3, 1, 2, 4, 1, 5};
      CHECK(result == correct);
    }

    SUBCASE("different data-type") {
      auto get_string_sequence = [](int x) -> std::vector<std::string> {
        return {
            std::to_string(x - 1), std::to_string(x), std::to_string(2 * x)};
      };

      std::vector<int> input = {2, 4, 10};
      std::vector<std::string> result = flatmap(input, get_string_sequence);
      std::vector<std::string> correct = {
          "1", "2", "4", "3", "4", "8", "9", "10", "20"};
      CHECK(result == correct);
    }
  }
}

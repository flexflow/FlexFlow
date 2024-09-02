#include "utils/containers/filtrans.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/set.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filtrans(std::vector<In>, F)") {
    std::vector<int> input = {1, 2, 3, 2, 4};
    std::vector<std::string> result =
        filtrans(input, [](int x) -> std::optional<std::string> {
          if ((x % 2) == 0) {
            return std::to_string(x);
          } else {
            return std::nullopt;
          }
        });

    std::vector<std::string> correct = {"2", "2", "4"};

    CHECK(result == correct);
  }

  TEST_CASE("filtrans(std::unordered_set<In>, F)") {
    std::unordered_set<int> input = {1, 2, 3, 4};
    std::unordered_set<std::string> result =
        filtrans(input, [](int x) -> std::optional<std::string> {
          if ((x % 2) == 0) {
            return std::to_string(x);
          } else {
            return std::nullopt;
          }
        });

    std::unordered_set<std::string> correct = {"2", "4"};

    CHECK(result == correct);
  }

  TEST_CASE("filtrans(std::set<In>, F)") {
    std::set<int> input = {1, 2, 3, 4};
    std::set<std::string> result =
        filtrans(input, [](int x) -> std::optional<std::string> {
          if ((x % 2) == 0) {
            return std::to_string(x);
          } else {
            return std::nullopt;
          }
        });

    std::set<std::string> correct = {"2", "4"};

    CHECK(result == correct);
  }
}

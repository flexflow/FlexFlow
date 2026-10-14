#include "utils/containers/transform_until.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform_until") {
    auto f = [](int x) -> std::optional<int> {
      if (x >= 0) {
        return x + 1;
      } else {
        return std::nullopt;
      }
    };

    SUBCASE("transforms full container") {
      std::vector<int> input = {2, 5, 1, 2, 0};

      std::vector<int> result = transform_until(input, f);
      std::vector<int> correct = {3, 6, 2, 3, 1};

      CHECK(result == correct);
    }

    SUBCASE("transforms none") {
      std::vector<int> input = {-1, 5, 1, 2, 0};

      std::vector<int> result = transform_until(input, f);
      std::vector<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("transforms part of container") {
      std::vector<int> input = {2, 5, -1, 2, 0};

      std::vector<int> result = transform_until(input, f);
      std::vector<int> correct = {3, 6};

      CHECK(result == correct);
    }

    SUBCASE("input container is empty") {
      std::vector<int> input = {};

      std::vector<int> result = transform_until(
          input, [](int x) -> std::optional<int> { PANIC("err"); });
      std::vector<int> correct = {};

      CHECK(result == correct);
    }
  }
}

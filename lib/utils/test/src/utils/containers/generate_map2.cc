#include "utils/containers/generate_map2.h"
#include "test/utils/doctest/fmt/map.h"
#include "test/utils/doctest/fmt/vector.h"
#include "utils/containers/range.h"
#include <doctest/doctest.h>
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("generate_map2") {
    SUBCASE("does not generate duplicate keys") {
      std::vector<int> input = {1, 3, 2};

      std::map<std::string, std::vector<int>> result = generate_map2(
          input,
          [&](int k) -> std::string { return std::to_string(k); },
          [&](int k) -> std::vector<int> { return range(k); });

      std::map<std::string, std::vector<int>> correct = {
          {"1", {0}},
          {"2", {0, 1}},
          {"3", {0, 1, 2}},
      };

      CHECK(result == correct);
    }

    SUBCASE("generates duplicates pairs") {
      std::vector<int> input = {1, 3, 2, 5};

      std::map<std::string, std::vector<int>> result = generate_map2(
          input,
          [&](int k) -> std::string { return std::to_string(k % 4); },
          [&](int k) -> std::vector<int> { return range(k % 4); });

      std::map<std::string, std::vector<int>> correct = {
          {"1", {0}},
          {"2", {0, 1}},
          {"3", {0, 1, 2}},
      };

      CHECK(result == correct);
    }

    SUBCASE("starts with duplicate keys") {
      std::vector<int> input = {1, 3, 2, 3};

      std::map<std::string, std::vector<int>> result = generate_map2(
          input,
          [&](int k) -> std::string { return std::to_string(k); },
          [&](int k) -> std::vector<int> { return range(k); });

      std::map<std::string, std::vector<int>> correct = {
          {"1", {0}},
          {"2", {0, 1}},
          {"3", {0, 1, 2}},
      };

      CHECK(result == correct);
    }

    SUBCASE("generates duplicate keys") {
      std::vector<int> input = {1, 3, 2, 5};

      CHECK_THROWS(generate_map2(
          input,
          [&](int k) -> std::string { return std::to_string(k % 4); },
          [&](int k) -> std::vector<int> { return range(k); }));
    }
  }
}

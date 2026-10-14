#include "utils/containers/merge_unordered_maps_with.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/binary_merge_unordered_maps_with.h"
#include <doctest/doctest.h>
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("merge_unordered_maps_with") {
    auto string_concat = [](std::string const &l, std::string const &r) {
      return l + r;
    };

    RC_SUBCASE(
        "with two inputs, matches binary_merge_unordered_maps_with",
        [&](std::unordered_map<int, std::string> const &lhs,
            std::unordered_map<int, std::string> const &rhs) {
          std::unordered_map<int, std::string> from_merge_unordered_maps_with =
              merge_unordered_maps_with(std::vector{lhs, rhs}, string_concat);

          std::unordered_map<int, std::string>
              from_binary_merge_unordered_maps_with =
                  binary_merge_unordered_maps_with(lhs, rhs, string_concat);

          CHECK(from_merge_unordered_maps_with ==
                from_binary_merge_unordered_maps_with);
        });

    SUBCASE("maps overlap") {
      std::unordered_map<int, std::string> map1 = {
          {1, "map1_one."},
          {4, "map1_four."},
      };

      std::unordered_map<int, std::string> map2 = {
          {2, "map2_two."},
          {4, "map2_four."},
          {5, "map2_five."},
      };

      std::unordered_map<int, std::string> map3 = {
          {1, "map3_one."},
      };

      std::unordered_map<int, std::string> result = merge_unordered_maps_with(
          std::vector{map1, map2, map3}, string_concat);

      std::unordered_map<int, std::string> correct = {
          {1, "map1_one.map3_one."},
          {2, "map2_two."},
          {4, "map1_four.map2_four."},
          {5, "map2_five."},
      };

      CHECK(result == correct);
    }

    auto fail_if_called = [](std::string const &,
                             std::string const &) -> std::string { PANIC(); };

    SUBCASE("maps do not overlap") {
      std::unordered_map<int, std::string> map1 = {
          {8, "map1_eight."},
          {4, "map1_four."},
      };

      std::unordered_map<int, std::string> map2 = {
          {2, "map2_two."},
          {5, "map2_five."},
      };

      std::unordered_map<int, std::string> map3 = {
          {1, "map3_one."},
      };

      std::unordered_map<int, std::string> result = merge_unordered_maps_with(
          std::vector{map1, map2, map3}, fail_if_called);

      std::unordered_map<int, std::string> correct = {
          {1, "map3_one."},
          {2, "map2_two."},
          {4, "map1_four."},
          {5, "map2_five."},
          {8, "map1_eight."},
      };

      CHECK(result == correct);
    }

    SUBCASE("no maps are provided") {
      std::vector<std::unordered_map<int, std::string>> maps = {};

      std::unordered_map<int, std::string> result =
          merge_unordered_maps_with(maps, fail_if_called);

      std::unordered_map<int, std::string> correct = {};

      CHECK(result == correct);
    }
  }
}

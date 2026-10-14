#include "utils/containers/merge_maps_with.h"
#include "test/utils/doctest/fmt/map.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/binary_merge_maps_with.h"
#include <doctest/doctest.h>
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("merge_maps_with") {
    auto string_concat = [](std::string const &l, std::string const &r) {
      return l + r;
    };

    RC_SUBCASE("with two inputs, matches binary_merge_maps_with",
               [&](std::map<int, std::string> const &lhs,
                   std::map<int, std::string> const &rhs) {
                 std::map<int, std::string> from_merge_maps_with =
                     merge_maps_with(std::vector{lhs, rhs}, string_concat);

                 std::map<int, std::string> from_binary_merge_maps_with =
                     binary_merge_maps_with(lhs, rhs, string_concat);

                 CHECK(from_merge_maps_with == from_binary_merge_maps_with);
               });

    SUBCASE("maps overlap") {
      std::map<int, std::string> map1 = {
          {1, "map1_one."},
          {4, "map1_four."},
      };

      std::map<int, std::string> map2 = {
          {2, "map2_two."},
          {4, "map2_four."},
          {5, "map2_five."},
      };

      std::map<int, std::string> map3 = {
          {1, "map3_one."},
      };

      std::map<int, std::string> result =
          merge_maps_with(std::vector{map1, map2, map3}, string_concat);

      std::map<int, std::string> correct = {
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
      std::map<int, std::string> map1 = {
          {8, "map1_eight."},
          {4, "map1_four."},
      };

      std::map<int, std::string> map2 = {
          {2, "map2_two."},
          {5, "map2_five."},
      };

      std::map<int, std::string> map3 = {
          {1, "map3_one."},
      };

      std::map<int, std::string> result =
          merge_maps_with(std::vector{map1, map2, map3}, fail_if_called);

      std::map<int, std::string> correct = {
          {1, "map3_one."},
          {2, "map2_two."},
          {4, "map1_four."},
          {5, "map2_five."},
          {8, "map1_eight."},
      };

      CHECK(result == correct);
    }

    SUBCASE("no maps are provided") {
      std::vector<std::map<int, std::string>> maps = {};

      std::map<int, std::string> result = merge_maps_with(maps, fail_if_called);

      std::map<int, std::string> correct = {};

      CHECK(result == correct);
    }
  }
}

#include "utils/containers/filtermap_values.h"
#include "test/utils/doctest.h"
#include "utils/fmt/unordered_map.h"
#include "utils/fmt/map.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filtermap_values(std::unordered_map<K, V>, F)") {
    std::unordered_map<int, std::string> input = {
        {1, "one"},
        {2, "two"},
    };
    std::unordered_map<int, int> result =
        filtermap_values(input, [](std::string const &v) -> std::optional<int> {
          if (v == "two") {
            return std::nullopt;
          } else {
            return v.size() + 1;
          }
        });
    std::unordered_map<int, int> correct = {
        {1, 4},
    };
    CHECK(result == correct);
  }

  TEST_CASE("filtermap_values(std::map<K, V>, F)") {
    std::map<int, std::string> input = {
        {1, "one"},
        {2, "two"},
    };
    std::map<int, int> result =
        filtermap_values(input, [](std::string const &v) -> std::optional<int> {
          if (v == "two") {
            return std::nullopt;
          } else {
            return v.size() + 1;
          }
        });
    std::map<int, int> correct = {
        {1, 4},
    };
    CHECK(result == correct);
  }
}

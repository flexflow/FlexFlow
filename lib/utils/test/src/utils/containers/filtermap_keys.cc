#include "utils/containers/filtermap_keys.h"
#include "test/utils/doctest/fmt/map.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filtermap_keys(std::unordered_map<K, V>, F)") {
    std::unordered_map<int, std::string> input = {
        {1, "one"},
        {2, "two"},
    };
    std::unordered_map<std::string, std::string> result =
        filtermap_keys(input, [](int k) -> std::optional<std::string> {
          if (k == 1) {
            return std::nullopt;
          } else {
            std::ostringstream oss;
            oss << (k + 1);
            return oss.str();
          }
        });
    std::unordered_map<std::string, std::string> correct = {
        {"3", "two"},
    };
    CHECK(result == correct);
  }

  TEST_CASE("filtermap_keys(std::map<K, V>, F)") {
    std::map<int, std::string> input = {
        {1, "one"},
        {2, "two"},
    };
    std::map<std::string, std::string> result =
        filtermap_keys(input, [](int k) -> std::optional<std::string> {
          if (k == 1) {
            return std::nullopt;
          } else {
            std::ostringstream oss;
            oss << (k + 1);
            return oss.str();
          }
        });
    std::map<std::string, std::string> correct = {
        {"3", "two"},
    };
    CHECK(result == correct);
  }
}

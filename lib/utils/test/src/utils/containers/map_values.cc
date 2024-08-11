#include "utils/containers/map_values.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_map>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("map_values") {
    std::unordered_map<int, std::string> m = {{1, "one"}, {2, "two"}};
    auto f = [](std::string const &s) { return s.size(); }; // Mapping function
    std::unordered_map<int, size_t> result = map_values(m, f);
    std::unordered_map<int, size_t> expected = {{1, 3}, {2, 3}};
    CHECK(result == expected);
  }
}

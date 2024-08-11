#include "utils/containers/map_keys.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_map>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("map_keys") {
    std::unordered_map<int, std::string> m = {{1, "one"}, {2, "two"}};
    auto f = [](int x) { return x * x; }; // Mapping function
    auto result = map_keys(m, f);
    CHECK(result.size() == 2);
    CHECK(result[1] == "one");
    CHECK(result[4] == "two");
  }
}

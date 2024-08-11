#include "utils/containers/filter_keys.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_map>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("filter_keys") {
    std::unordered_map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}};
    auto f = [](int x) { return x % 2 == 1; }; // Filtering function
    std::unordered_map<int, std::string> result = filter_keys(m, f);
    std::unordered_map<int, std::string> expected = {{1, "one"}, {3, "three"}};
    CHECK(result == expected);
  }
}

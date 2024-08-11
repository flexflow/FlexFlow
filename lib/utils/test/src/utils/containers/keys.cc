#include "utils/containers/keys.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("keys") {
    std::unordered_map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}};
    std::unordered_set<int> result = keys(m);
    std::unordered_set<int> expected = {3, 2, 1};
    CHECK(result == expected);
  }
}

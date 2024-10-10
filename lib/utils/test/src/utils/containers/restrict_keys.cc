#include "utils/containers/restrict_keys.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <doctest/doctest.h>
#include <string>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("restrict_keys") {
    std::unordered_map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}};
    std::unordered_set<int> mask = {2, 3, 4};
    std::unordered_map<int, std::string> result = restrict_keys(m, mask);
    std::unordered_map<int, std::string> correct = {{2, "two"}, {3, "three"}};
    CHECK(result == correct);
  }
}

#include "utils/containers/values.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("values") {
    std::unordered_map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}};
    std::vector<std::string> result = values(m);
    std::vector<std::string> expected = {"three", "two", "one"};
    CHECK(result == expected);
  }
}

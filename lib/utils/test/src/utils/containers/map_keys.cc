#include "utils/containers/map_keys.h"
#include "utils/fmt/unordered_map.h"
#include <doctest/doctest.h>
#include <string>
#include <unordered_map>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("map_keys") {
    SUBCASE("Distinct keys after transformation") {
      std::unordered_map<int, std::string> m = {{1, "one"}, {2, "two"}};
      auto f = [](int x) { return x * x; };
      std::unordered_map<int, std::string> result = map_keys(m, f);
      std::unordered_map<int, std::string> correct = {{1, "one"}, {4, "two"}};
      CHECK(correct == result);
    }

    SUBCASE("Non-distinct keys after transformation") {
      std::unordered_map<int, std::string> m = {
          {1, "one"}, {2, "two"}, {-1, "minus one"}};
      auto f = [](int x) { return std::abs(x); };
      CHECK_THROWS(map_keys(m, f));
    }
  }
}

#include "utils/containers/contains_key.h"
#include <doctest/doctest.h>
#include <map>
#include <unordered_map>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("contains_key(std::unordered_map<K, V>, K)") {
    std::unordered_map<int, std::string> m = {
        {1, "one"},
    };
    CHECK(contains_key(m, 1));
    CHECK_FALSE(contains_key(m, 2));
  }

  TEST_CASE("contains_key(std::map<K, V>, K)") {
    std::map<int, std::string> m = {
        {1, "one"},
    };
    CHECK(contains_key(m, 1));
    CHECK_FALSE(contains_key(m, 2));
  }
}

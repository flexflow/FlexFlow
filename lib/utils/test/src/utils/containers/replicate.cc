#include "utils/containers/replicate.h"
#include "utils/hash/unordered_set.h"
#include <doctest/doctest.h>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("replicate") {
    SUBCASE("ints") {
      int x = 42;
      std::vector<int> result = replicate(5, x);
      std::vector<int> correct = {42, 42, 42, 42, 42};
      CHECK(result == correct);
    }
    SUBCASE("unordered_set") {
      std::unordered_set<float> x = {1.0, 1.5};
      std::vector<std::unordered_set<float>> result = replicate(3, x);
      std::vector<std::unordered_set<float>> correct = {
          {1.0, 1.5}, {1.0, 1.5}, {1.0, 1.5}};
      CHECK(result == correct);
    }
  }
}

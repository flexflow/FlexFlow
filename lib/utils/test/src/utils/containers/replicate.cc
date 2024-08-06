#include "utils/containers/replicate.h"
#include "utils/hash/unordered_set.h"
#include <doctest/doctest.h>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("replicate") {
    SUBCASE("ints") {
      int x = 42;
      std::unordered_multiset<int> result = replicate(5, x);
      std::unordered_multiset<int> correct = {42, 42, 42, 42, 42};
      CHECK(result == correct);
    }
    SUBCASE("unordered_set") {
      std::unordered_set<float> x = {1.0, 1.5};
      std::unordered_multiset<std::unordered_set<float>> result =
          replicate(3, x);
      std::unordered_multiset<std::unordered_set<float>> correct = {
          {1.0, 1.5}, {1.0, 1.5}, {1.0, 1.5}};
      CHECK(result == correct);
    }
  }
}

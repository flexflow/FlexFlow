#include <doctest/doctest.h>
#include "utils/containers/enumerate.h"
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("enumerate(std::vector<T>)") {
    std::vector<std::string> input = {"zero", "one", "two", "three"};

    std::map<int, std::string> correct = {
      {0, "zero"},
      {1, "one"},
      {2, "two"},
      {3, "three"},
    };
  }

  TEST_CASE("enumerate(std::unordered_set<T>)") {
    std::unordered_set<std::string> input = {"zero", "one", "two", "three"};

    std::map<int, std::string> correct = {
      {0, "zero"},
      {1, "one"},
      {2, "two"},
      {3, "three"},
    };
  }
}

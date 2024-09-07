#include "utils/containers/maximum.h"
#include <doctest/doctest.h>
#include <vector>
#include <unordered_set>
#include <set>
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/doctest/fmt/set.h"
#include "test/utils/doctest/fmt/multiset.h"
#include "test/utils/doctest/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("maximum(T)", T, std::vector<int>, 
                                      std::unordered_set<int>,
                                      std::unordered_multiset<int>,
                                      std::set<int>,
                                      std::multiset<int>) {
    
    SUBCASE("input is empty") {
      T input = {};

      std::optional<int> result = maximum(input);
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input does not have duplicates") {
      T input = {1, 3, 2};

      std::optional<int> result = maximum(input);
      std::optional<int> correct = 3;

      CHECK(result == correct);
    }

    SUBCASE("input has duplicates") {
      T input = {1, 2, 2, 0};

      std::optional<int> result = maximum(input);
      std::optional<int> correct = 2;

      CHECK(result == correct);
    }
  }

  TEST_CASE("maximum(std::vector<std::string>)") {
    std::vector<std::string> input = {"hello", "world"};

    std::optional<std::string> result = maximum(input);
    std::optional<std::string> correct = "world";

    CHECK(result == correct);
  }
}

#include "utils/fmt/unordered_set.h"
#include "test/utils/doctest.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/fmt/unordered_multiset.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::unordered_set<int>)") {
    std::unordered_set<int> input = {0, 1, 3, 2};
    std::string result = fmt::to_string(input);
    std::string correct = "{0, 1, 2, 3}";
    std::unordered_multiset<char> unordered_result =
        unordered_multiset_of(result);
    std::unordered_multiset<char> unordered_correct =
        unordered_multiset_of(correct);
    CHECK(unordered_result == unordered_correct);
  }
}

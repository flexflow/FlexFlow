#include "utils/fmt/unordered_set.h"
#include "test/utils/doctest.h"
#include "utils/containers/get_element_counts.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::unordered_set<int>)") {
    std::unordered_set<int> input = {0, 1, 3, 2};
    std::string result = fmt::to_string(input);
    std::string correct = "{0, 1, 2, 3}";
    std::unordered_map<char, int> result_char_counts =
        get_element_counts(result);
    std::unordered_map<char, int> correct_char_counts =
        get_element_counts(correct);
    CHECK(result_char_counts == correct_char_counts);
  }
}

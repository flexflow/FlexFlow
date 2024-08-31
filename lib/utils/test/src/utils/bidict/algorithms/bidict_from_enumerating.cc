#include <doctest/doctest.h>
#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("bidict_from_enumerating(std::unordered_set<T>)") {
    std::unordered_set<std::string> input = {"zero", "one", "two"};

    bidict<int, std::string> result = bidict_from_enumerating(input);

    std::unordered_set<int> result_left_entries = left_entries(result);
    std::unordered_set<int> correct_left_entries = {0, 1, 2};
    CHECK(result_left_entries == correct_left_entries);

    std::unordered_set<std::string> result_right_entries = right_entries(result);
    std::unordered_set<std::string> correct_right_entries = input;
    CHECK(result_right_entries == correct_right_entries);
  }

  TEST_CASE("bidict_from_enumerating(std::set<T>)") {
    std::set<std::string> input = {"a", "c", "b"};

    bidict<int, std::string> correct = {
      {0, "a"},
      {1, "b"},
      {2, "c"},
    };

    bidict<int, std::string> result = bidict_from_enumerating(input);

    CHECK(result == correct);
  }
}

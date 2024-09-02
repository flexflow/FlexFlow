#include "utils/containers/inplace_filter.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/map.h"
#include "test/utils/doctest/fmt/set.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("inplace_filter(T, F)",
                     T,
                     std::vector<int>,
                     std::unordered_set<std::string>,
                     std::set<std::string>,
                     std::unordered_map<int, int>,
                     std::map<int, std::string>) {
    RC_SUBCASE("inplace_filter returns empty for predicate always_false",
               [](T t) {
                 auto always_false = [](auto const &) { return false; };
                 inplace_filter(t, always_false);
                 return t.size() == 0;
               });

    RC_SUBCASE("inplace_filter returns input for predicate always_true",
               [](T t) {
                 T input = t;
                 auto always_true = [](auto const &) { return true; };
                 inplace_filter(t, always_true);
                 return t == input;
               });
  }

  TEST_CASE("inplace_filter(std::vector &, F)") {
    std::vector<int> input = {1, 2, 3, 4, 5};
    auto predicate = [](int x) { return x % 2 == 0; };

    inplace_filter(input, predicate);
    std::vector<int> correct = {2, 4};
    CHECK(input == correct);
  }

  TEST_CASE("inplace_filter(std::unordered_set &, F)") {
    std::unordered_set<int> input = {1, 2, 3, 4, 5, 6, 7, 8};
    auto predicate = [](int x) { return x % 2 == 0; };

    inplace_filter(input, predicate);
    std::unordered_set<int> correct = {2, 4, 6, 8};
    CHECK(input == correct);
  }

  TEST_CASE("inplace_filter(std::set &, F)") {
    std::set<int> input = {3, 2, 5, 8};
    auto predicate = [](int x) { return x % 2 == 0; };

    inplace_filter(input, predicate);
    std::set<int> correct = {2, 8};
    CHECK(input == correct);
  }

  TEST_CASE("inplace_filter(std::unordered_map &, F)") {
    std::unordered_map<int, std::string> input = {
        {3, "4"},
        {1, "1"},
        {2, "9"},
        {4, "4"},
    };
    auto predicate = [](std::pair<int, std::string> const &x) {
      return std::to_string(x.first) == x.second;
    };

    inplace_filter(input, predicate);
    std::unordered_map<int, std::string> correct = {
        {1, "1"},
        {4, "4"},
    };
    CHECK(input == correct);
  }

  TEST_CASE("inplace_filter(std::map &, F)") {
    std::map<int, std::string> input = {
        {3, "4"},
        {1, "1"},
        {2, "9"},
        {4, "4"},
    };
    auto predicate = [](std::pair<int, std::string> const &x) {
      return std::to_string(x.first) != x.second;
    };

    inplace_filter(input, predicate);
    std::map<int, std::string> correct = {
        {3, "4"},
        {2, "9"},
    };
    CHECK(input == correct);
  }
}

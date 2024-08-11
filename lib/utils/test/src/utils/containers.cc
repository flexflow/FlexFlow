#include "utils/containers.h"
#include "test/utils/doctest.h"
#include "utils/bidict/bidict.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("sum") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    CHECK(sum(v) == 15);
  }

  TEST_CASE("sum_where") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto condition = [](int x) { return x % 2 == 0; };
    CHECK(sum_where(v, condition) == 6);
  }

  TEST_CASE("product_where") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto condition = [](int x) { return x % 2 == 0; };
    CHECK(product_where(v, condition) == 8);
  }

  TEST_CASE("contains_l and contains_r") {
    bidict<int, std::string> bd;
    bd.equate(1, "one");
    bd.equate(2, "two");

    CHECK(contains_l(bd, 1) == true);
    CHECK(contains_l(bd, 3) == false);
    CHECK(contains_r(bd, std::string("one")) == true);
    CHECK(contains_r(bd, std::string("three")) == false);
  }

  TEST_CASE("is_submap") {
    std::unordered_map<int, std::string> m1 = {
        {1, "one"}, {2, "two"}, {3, "three"}};
    std::unordered_map<int, std::string> m2 = {{1, "one"}, {2, "two"}};
    std::unordered_map<int, std::string> m3 = {{1, "one"}, {4, "four"}};

    CHECK(is_submap(m1, m2) == true);
    CHECK(is_submap(m1, m3) == false);
  }

  TEST_CASE("index_of") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    CHECK(index_of(v, 3).value() == 2);
    CHECK(index_of(v, 6) == std::nullopt);
  }

  TEST_CASE("merge_maps") {
    SUBCASE("bidict") {
      std::unordered_map<int, std::string> fwd_map1 = {{1, "one"}, {2, "two"}};
      std::unordered_map<std::string, int> bwd_map1 = {{"one", 1}, {"two", 2}};
      std::unordered_map<int, std::string> fwd_map2 = {{3, "three"},
                                                       {4, "four"}};
      std::unordered_map<std::string, int> bwd_map2 = {{"three", 3},
                                                       {"four", 4}};
      bidict<int, std::string> lhs{fwd_map1, bwd_map1};
      bidict<int, std::string> rhs{fwd_map2, bwd_map2};

      std::unordered_map<int, std::string> result =
          merge_maps(lhs, rhs); // impicit conversion
      std::unordered_map<int, std::string> expected = {
          {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};
      CHECK(result == expected);
    }
    SUBCASE("unordered_map") {
      std::unordered_map<int, std::string> lhs = {{1, "one"}, {2, "two"}};
      std::unordered_map<int, std::string> rhs = {{3, "three"}, {4, "four"}};
      std::unordered_map<int, std::string> result = merge_maps(lhs, rhs);
      std::unordered_map<int, std::string> expected = {
          {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};
    }
  }

  TEST_CASE("lookup_in") {
    std::unordered_map<int, std::string> m = {{1, "one"}, {2, "two"}};
    auto f = lookup_in(m);
    CHECK(f(1) == "one");
    CHECK(f(2) == "two");
  }

  TEST_CASE("lookup_in_l") {
    bidict<int, std::string> m;
    m.equate(1, "one");
    m.equate(2, "two");
    auto f = lookup_in_l(m);
    CHECK(f(1) == "one");
    CHECK(f(2) == "two");
  }

  TEST_CASE("lookup_in_r") {
    bidict<int, std::string> m;
    m.equate(1, "one");
    m.equate(2, "two");
    auto f = lookup_in_r(m);
    CHECK(f("one") == 1);
    CHECK(f("two") == 2);
  }

  TEST_CASE("is_superseteq_of") {
    std::unordered_set<int> s1 = {1, 2, 3, 4};
    std::unordered_set<int> s2 = {1, 2, 3};
    std::unordered_set<int> s3 = {1, 2, 5};

    CHECK(is_superseteq_of(s1, s2) == true);
    CHECK(is_superseteq_of(s1, s3) == false);
  }

  TEST_CASE("restrict_keys") {
    std::unordered_map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}};
    std::unordered_set<int> mask = {2, 3, 4};
    std::unordered_map<int, std::string> result = restrict_keys(m, mask);
    std::unordered_map<int, std::string> expected = {{2, "two"}, {3, "three"}};
    CHECK(result == expected);
  }

  TEST_CASE("optional_all_of") {
    std::vector<int> v = {2, 4, 6, 8};
    auto f = [](int x) -> std::optional<bool> { return x % 2 == 0; };
    CHECK(optional_all_of(v, f) == true);

    auto f2 = [](int x) -> std::optional<bool> {
      if (x == 6) {
        return std::nullopt;
      }
      return x % 2 == 0;
    };
    CHECK(optional_all_of(v, f2) == std::nullopt);
  }

  TEST_CASE("are_all_same") {
    std::vector<int> v1 = {2, 2, 2, 2};
    std::vector<int> v2 = {1, 2, 3, 4};
    CHECK(are_all_same(v1) == true);
    CHECK(are_all_same(v2) == false);
  }

  TEST_CASE("Test for flatmap function on vectors") {

    auto get_factors = [](int x) -> std::vector<int> {
      // Returns a vector of factors of x
      std::vector<int> factors;
      for (int i = 1; i <= x; i++) {
        if (x % i == 0) {
          factors.push_back(i);
        }
      }
      return factors;
    };

    std::vector<int> v = {2, 3, 4, 5};
    auto result = flatmap<int, decltype(get_factors), int>(v, get_factors);
    CHECK(result == std::vector<int>({1, 2, 1, 3, 1, 2, 4, 1, 5}));
  }

  TEST_CASE("compare_by") {
    std::vector<std::string> v = {"abc", "a", "ab"};
    auto comp = compare_by<std::string>(
        [](std::string const &s) { return s.length(); });
    std::sort(v.begin(), v.end(), comp);
    CHECK(v == std::vector<std::string>{"a", "ab", "abc"});
  }

  TEST_CASE("maximum") {
    std::vector<int> v = {1, 5, 3, 4, 2};
    CHECK(maximum(v) == 5);
  }

  TEST_CASE("value_all") {
    std::vector<std::optional<int>> v = {1, 2, std::nullopt, 4, 5};
    CHECK_THROWS_AS(value_all(v), std::runtime_error);

    std::vector<std::optional<int>> v2 = {1, 2, 3, 4, 5};
    CHECK(value_all(v2) == std::vector<int>{1, 2, 3, 4, 5});
  }
}

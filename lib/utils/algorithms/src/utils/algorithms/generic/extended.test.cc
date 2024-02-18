#include "utils/testing.h"
#include "utils/algorithms/generic/extended.h"
#include "utils/rapidcheck_extra.h"
#include "utils/rapidcheck_extra/optional.h"
#include <utility>

/* TEST_SUITE(FF_TEST_SUITE) { */
/*   TEST_CASE_TEMPLATE("extended(std::vector, _)", */ 
/*                      T, */ 
/*                      std::vector<int>, */
/*                      std::list<int>, */
/*                      std::array<int, 10>) { */
/*     using Elem = typename T::value_type; */

/*     rc::dc_check("size(result) = size(in1) + size(in2)", [&](std::vector<Elem> const &lhs, T const &rhs) { */
/*       CHECK(extended(lhs, rhs).size() == lhs.size() + rhs.size()); */
/*     }); */

/*     rc::dc_check("preserves elements and order", [&](std::vector<Elem> const &lhs, T const &rhs) { */
/*       std::vector<Elem> result = extended(lhs, rhs); */

/*       std::vector<Elem> prefix = {result.cbegin(), result.cbegin() + lhs.size()}; */
/*       std::vector<Elem> correct_prefix = {lhs.cbegin(), lhs.cend()}; */
/*       CHECK(prefix == correct_prefix); */

/*       std::vector<Elem> postfix = {result.cbegin() + lhs.size(), result.cend()}; */
/*       std::vector<Elem> correct_postfix = {rhs.cbegin(), rhs.cend()}; */
/*       CHECK(postfix == correct_postfix); */
/*     }); */
/*   } */
  
/*   TEST_CASE_TEMPLATE("extended(std::unordered_set, _)", */ 
/*                      T, */ 
/*                      std::vector<int>, */
/*                      std::unordered_set<int>) { */
/*     using Elem = std::decay_t<typename T::value_type>; */

/*     rc::dc_check("result = union(lhs, rhs)", [&](std::unordered_set<Elem> const &lhs, T const &rhs) { */
/*       std::vector<Elem> vec_expected; */
/*       std::set_union(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(), std::back_inserter(vec_expected)); */
/*       std::unordered_set<Elem> expected_result = { vec_expected.cbegin(), vec_expected.cend() }; */
/*       CHECK(extended(lhs, rhs) == expected_result); */
/*     }); */
/*   } */

/*   TEST_CASE_TEMPLATE("extended(_, optional)", */
/*                      T, */
/*                      std::vector<int>, */
/*                      std::unordered_set<int>) { */
/*     using Elem = std::decay_t<typename T::value_type>; */

/*     rc::dc_check("result = rhs.has_value() ? extend(...) : lhs", [&](T const &lhs, std::optional<Elem> const &rhs) { */
/*       if (rhs.has_value()) { */
/*         CHECK(extended(lhs, rhs) == extended(lhs, std::vector{rhs.value()})); */
/*       } else { */
/*         CHECK(extended(lhs, rhs) == lhs); */
/*       } */
/*     }); */
/*   } */

/*   TEST_CASE("extended(_) examples") { */
/*     std::vector<int> vec = {1, 2, 3}; */
/*     std::unordered_set<int> set = {3, 4, 5}; */
/*     std::optional<int> opt = 2; */

/*     CHECK(extended(vec, vec) == std::vector{1, 2, 3, 1, 2, 3}); */
/*     CHECK(extended(set, set) == set); */
/*     CHECK(extended(set, vec) == std::unordered_set{1, 2, 3, 4, 5}); */
/*     CHECK(extended(vec, opt) == std::vector{1, 2, 3, 2}); */
/*     CHECK(extended(vec, std::nullopt) == vec); */
/*     CHECK(extended(set, opt) == std::unordered_set{2, 3, 4, 5}); */
/*     CHECK(extended(set, std::nullopt) == set); */
/*   } */
/* } */

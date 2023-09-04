#include "test/utils/all.h"
#include "utils/required.h"

using namespace FlexFlow;

TEST_CASE_TEMPLATE(
    "required supports expected operations", T, int, float, double, char) {
  rc::dc_check("static_cast",
               [&](T t) { CHECK(static_cast<T>(required<T>{t}) == t); });

  rc::dc_check("ops", [&](T t1, T t2) {
    required<T> r1{t1};
    required<T> r2{t2};

    auto check_overloads = [&](auto const &f) {
      CHECK(f(r1, r2) == f(t1, t2));
      CHECK(f(t1, r2) == f(t1, t2));
      CHECK(f(r1, t2) == f(t1, t2));
    };

    check_overloads([](auto const &lhs, auto const &rhs) { return lhs + rhs; });
    check_overloads([](auto const &lhs, auto const &rhs) { return lhs - rhs; });
    check_overloads([](auto const &lhs, auto const &rhs) { return lhs * rhs; });
  });
}

TEST_CASE_TEMPLATE("required supports expected operations (generic)",
                   T,
                   std::vector<int>,
                   std::string,
                   int,
                   float) {
  CHECK_FALSE(std::is_default_constructible_v<required_wrapper_impl<T>>);

  rc::dc_check("static_cast",
               [&](T t) { CHECK(static_cast<T>(required<T>{t}) == t); });

  rc::dc_check("equals", [&](T t) { CHECK(required<T>{t} == t); });

  rc::dc_check("copy constructible", [&](T t) {
    required<T> r{t};
    required<T> r2(r);
    CHECK(r == r2);
  });

  rc::dc_check("copy assignable", [&](T t) {
    required<T> r{t};
    required<T> r2 = r;
    CHECK(r == r2);
  });

  rc::dc_check("move constructible", [&](T t) {
    required<T> r{t};
    required<T> r2{t};
    required<T> r3{std::move(r2)};
    CHECK(r == r3);
  });

  rc::dc_check("move assignable", [&](T t) {
    required<T> r{t};
    required<T> r2{t};
    required<T> r3 = std::move(r2);
    CHECK(r == r3);
  });

  REQUIRE(is_fmtable_v<T>);
  rc::dc_check("fmtable", [&](T t) {
    required<T> r{t};
    CHECK(fmt::to_string(r) == fmt::to_string(t));
  });
}

TEST_CASE("required supports expected operations (std::vector<int>)") {
  std::vector<int> t;
  required<std::vector<int>> r = std::vector<int>{};

  t.push_back(4);
  r.push_back(4);
  CHECK(t == r);
  CHECK(t.size() == r.size());
  CHECK(t.at(0) == r.at(0));
  CHECK(t[0] == r[0]);
}

struct castable_to_int {
  operator int() const;
};

TEST_CASE_TEMPLATE("required supports expected casts",
                   T,
                   std::pair<int, double>,
                   std::pair<castable_to_int, int>) {
  using LHS = typename T::first_type;
  using RHS = typename T::second_type;
  REQUIRE(std::is_convertible_v<LHS, RHS>);
  CHECK(std::is_convertible_v<required<LHS>, RHS>);
}

TEST_CASE_TEMPLATE("required does not support unexpected casts",
                   T,
                   std::pair<int, std::string>,
                   std::pair<castable_to_int, std::unordered_set<bool>>) {
  using LHS = typename T::first_type;
  using RHS = typename T::second_type;

  REQUIRE(!std::is_convertible_v<LHS, RHS>);
  CHECK(!std::is_convertible_v<required<LHS>, RHS>);
}

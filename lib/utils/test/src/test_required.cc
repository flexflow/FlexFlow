#include "test/utils/doctest.h"
#include "utils/required.h"
#include "test/utils/rapidcheck/doctest.h"

using namespace FlexFlow;

struct default_constructible_example {
  int member;

  default_constructible_example()
    : member(1234)
  { }

  friend bool operator==(default_constructible_example const &lhs,
                         default_constructible_example const &rhs) {
    return lhs.member == rhs.member;
  }

  friend bool operator!=(default_constructible_example const &lhs,
                         default_constructible_example const &rhs) {
    return lhs.member != rhs.member;
  }
};

TEST_CASE_TEMPLATE("required_wrapper_impl is not default constructible", T, int, float, double, char) {
  STATIC_CHECK(std::is_default_constructible<T>::value);
  STATIC_CHECK_FALSE(std::is_default_constructible<required_wrapper_impl<T>>::value);
}

TEST_CASE_TEMPLATE("required_inheritance_impl is not default constructible", 
                   T, 
                   std::vector<std::vector<int>>,
                   std::unordered_map<std::string, float>) {
  STATIC_CHECK(std::is_default_constructible<T>::value);
  STATIC_CHECK_FALSE(std::is_default_constructible<required_wrapper_impl<T>>::value);
}

TEST_CASE_TEMPLATE("required_wrapper_impl supports expected operations", T, int, float, double, char) {
  rc::dc_check("copy constructible", [&](T t) {
    required_wrapper_impl<T> r{t};
    required_wrapper_impl<T> r2(r);
    CHECK(r == r2);
  });

  rc::dc_check("static_cast", [&](T t) {
    CHECK(static_cast<T>(required_wrapper_impl<T>{t}) == t);
  });

  /* rc::dc_check("ops", [&](T t1, T t2) { */
  /*   required_wrapper_impl<T> r1{t1}; */
  /*   required_wrapper_impl<T> r2{t2}; */
  /*   STATIC_CHECK(std::is_same<decltype(std::declval<required_wrapper_impl<T>>() + std::declval<required_wrapper_impl<T>>()), required_wrapper_impl<decltype(t1 + t2)>>::value); */
  /*   CHECK(static_cast<T>(r1 + r2) == t1 + t2); */
  /*   CHECK(static_cast<T>(r1 - r2) == t1 - t2); */
  /*   CHECK(static_cast<T>(r1 * r2) == t1 * t2); */
  /* }); */
}

TEST_CASE_TEMPLATE("required_inheritance_impl supports expected operations (generic)",
                   T, 
                   std::vector<int>,
                   std::string) {
  rc::dc_check("copy constructible", [&](T t) {
    required_inheritance_impl<T> r{t};
    required_inheritance_impl<T> r2(r);
    assert(r == r2);
  });

  rc::dc_check("copy assignable", [&](T t) {
    required_inheritance_impl<T> r{t};
    required_inheritance_impl<T> r2 = r;
    assert(r == r2);
  });

  rc::dc_check("move constructible", [&](T t) {
    required_inheritance_impl<T> r{t};
    required_inheritance_impl<T> r2{t};
    required_inheritance_impl<T> r3{std::move(r2)};
    assert(r == r3);
  });

  rc::dc_check("move assignable", [&](T t) {
    required_inheritance_impl<T> r{t};
    required_inheritance_impl<T> r2{t};
    required_inheritance_impl<T> r3 = std::move(r2);
    assert(r == r3);
  });
}


TEST_CASE("required_inheritance_impl supports expected operations (std::vector<int>)") {
}

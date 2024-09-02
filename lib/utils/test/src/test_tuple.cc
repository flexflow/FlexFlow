#include <doctest/doctest.h>
#include "utils/tuple.h"

#include <iostream>
#include <utility>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get function") {
    std::tuple<int, float, double> t(42, 3.14f, 2.71828);

    SUBCASE("get mutable reference") {
      int &result = get<int>(t);
      CHECK(result == 42);

      result = 100;
      CHECK(std::get<0>(t) == 100);
    }

    SUBCASE("get rvalue reference") {
      int &&result = get<int>(std::move(t));
      CHECK(result == 42);

      // t is in a valid but unspecified state after move
      CHECK(std::get<0>(t) == 42); // Uncomment this line to check the behavior
    }

    SUBCASE("get const reference") {
      int const &result = get<int>(t);
      CHECK(result == 42);
    }

    SUBCASE("get const rvalue reference") {
      int const &&result = get<int>(std::move(t));
      CHECK(result == 42);
    }
  }

  TEST_CASE("tuple_prepend function") {
    std::tuple<float, double> t1(3.14f, 2.71828);
    int value = 42;

    auto result = tuple_prepend(value, t1);
    std::tuple<int, float, double> expected(42, 3.14f, 2.71828);
    CHECK(result == expected);
  }

  TEST_CASE("Testing tuple_head_t") {
    CHECK(std::is_same<tuple_head_t<1, std::tuple<int, float>>,
                       std::tuple<int>>::value);
    CHECK(std::is_same<tuple_head_t<0, std::tuple<int, float>>,
                       std::tuple<>>::value);
  }

  TEST_CASE("Testing tuple_slice_t") {
    CHECK(std::is_same<tuple_slice_t<0, 1, std::tuple<int, float, double>>,
                       std::tuple<int>>::value);
    CHECK(std::is_same<tuple_slice_t<-2, -1, std::tuple<int, float, double>>,
                       std::tuple<float>>::value);
    CHECK(std::is_same<tuple_slice_t<1, 3, std::tuple<int, float, double>>,
                       std::tuple<float, double>>::value);
  }

  TEST_CASE("Testing tuple_compare function") {
    std::tuple<int, double, char> tup1{1, 3.14, 'a'};
    std::tuple<int, double, char> tup2{1, 3.14, 'a'};
    std::tuple<int, double, char> tup3{2, 3.14, 'b'};

    CHECK(tuple_compare(tup1, tup2));
    CHECK(!tuple_compare(tup1, tup3));
  }

  TEST_CASE("Testing get function with valid index") {
    std::tuple<int, double, char> tup{1, 3.14, 'a'};

    CHECK(get<int>(tup) == 1);
    CHECK(get<double>(tup) == 3.14);
    CHECK(get<char>(tup) == 'a');
  }
}

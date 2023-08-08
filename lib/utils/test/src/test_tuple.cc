#include "doctest/doctest.h"
#include "utils/tuple.h"

#include <iostream>
#include <utility>

using namespace FlexFlow;

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

struct Visitor {
  template <typename T>
  void operator()(int idx, T const &value) {
    std::cout << "Value at index " << idx << ": " << value << std::endl;
  }
};

TEST_CASE("tuple_prepend function") {
  std::tuple<float, double> t1(3.14f, 2.71828);
  int value = 42;

  auto result = tuple_prepend(value, t1);
  std::tuple<int, float, double> expected(42, 3.14f, 2.71828);
  CHECK(result == expected);
}

TEST_CASE("Testing tuple_head_t") {
  CHECK(std::is_same<tuple_head_t<1, std::tuple<int, float>>, std::tuple<int>>::value);
  CHECK(std::is_same<tuple_head_t<0, std::tuple<int, float>>, std::tuple<>>::value);
}

TEST_CASE("Testing tuple_slice_t") {
  CHECK(std::is_same<tuple_slice_t<0, 1, std::tuple<int, float, double>>, std::tuple<int>>::value);
  CHECK(std::is_same<tuple_slice_t<-2, -1, std::tuple<int, float, double>>, std::tuple<float>>::value);
  CHECK(std::is_same<tuple_slice_t<1, 3, std::tuple<int, float, double>>, std::tuple<float, double>>::value);
}

TEST_CASE("Testing get function with invalid index") {
  std::tuple<int, float, double> tup(42, 3.14f, 2.71);
  CHECK_THROWS_AS(get<int>(tup), std::exception);
}

// TEST_CASE("tuple_slice_t function") {
//   std::tuple<int, float, double, char> t(42, 3.14f, 2.71828, 'A');

//   SUBCASE("tuple_head_t") {
//     using ResultType = tuple_head_t<2, decltype(t)>;
//     std::tuple<int, float> expected(42, 3.14f);
//     CHECK(std::is_same<ResultType, decltype(expected)>::value);
//     auto result =  tuple_head_t<2, decltype(t)>();
//     std::cout << "res:"<<std::get<0>(result) << ", " << std::get<1>(result)
//     << std::endl;

//     CHECK(tuple_head_t<2, decltype(t)>() == expected);

//     CHECK(tuple_compare(tuple_head_t<2, decltype(t)>(), expected));
//   }

//   SUBCASE("tuple_tail_t") {
//     using ResultType = tuple_tail_t<2, decltype(t)>;
//     std::tuple<double, char> expected(2.71828, 'A');
//     CHECK(std::is_same<ResultType, decltype(expected)>::value);
//     CHECK(tuple_compare(tuple_tail_t<2, decltype(t)>(), expected));
//   }

//   SUBCASE("tuple_slice_t") {
//     using ResultType = tuple_slice_t<1, 3, decltype(t)>;
//     std::tuple<float, double> expected(3.14f, 2.71828);
//     CHECK(std::is_same<ResultType, decltype(expected)>::value);
//     CHECK(tuple_slice_t<1, 3,decltype(t)>() == expected);
//   }
// }

// TEST_CASE("get function with invalid index") {
//   std::tuple<int, float, double> t(42, 3.14f, 2.71828);

//   SUBCASE("negative index") {
//     CHECK_THROWS_AS(get<int>(t, -1), std::runtime_error);
//   }

//   SUBCASE("index out of bounds") {
//     CHECK_THROWS_AS(get<int>(t, 3), std::runtime_error);
//   }
// }

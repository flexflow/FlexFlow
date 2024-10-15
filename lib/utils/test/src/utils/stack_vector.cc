#include "utils/stack_vector.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>
#include <iterator>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE(
      "stack_vector<T, MAXSIZE>::push_back", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector;

    vector.push_back(10);
    std::vector<T> result = vector;
    std::vector<T> correct = {10};
    CHECK(result == correct);

    vector.push_back(20);
    correct = {10, 20};
    result = vector;
    CHECK(result == correct);
  }

  TEST_CASE_TEMPLATE(
      "stack_vector<T, MAXSIZE>::operator[]", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector;

    vector.push_back(10);
    vector.push_back(20);
    vector.push_back(30);

    CHECK(vector[0] == 10);
    CHECK(vector[1] == 20);
    CHECK(vector[2] == 30);
  }

  TEST_CASE_TEMPLATE("stack_vector<T, MAXSIZE>::size", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector;

    CHECK(vector.size() == 0);

    vector.push_back(10);
    CHECK(vector.size() == 1);

    vector.push_back(20);
    CHECK(vector.size() == 2);
  }

  TEST_CASE_TEMPLATE(
      "stack_vector<T, MAXSIZE>::operator==", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector1, vector2;

    vector1.push_back(10);
    vector1.push_back(15);
    vector1.push_back(20);

    vector2.push_back(10);
    vector2.push_back(15);
    vector2.push_back(20);

    CHECK(vector1 == vector2);
  }

  TEST_CASE_TEMPLATE("stack_vector<T, MAXSIZE>::back", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackVector = stack_vector<T, MAXSIZE>;
    StackVector vector;

    vector.push_back(10);
    CHECK(vector.back() == 10);

    vector.push_back(20);
    CHECK(vector.back() == 20);
  }

  TEST_CASE_TEMPLATE(
      "stack_vector<T, MAXSIZE> - check for size bound", T, int, double, char) {
    constexpr std::size_t MAXSIZE = 10;
    RC_SUBCASE("within bound", [&](stack_vector<T, MAXSIZE> v) {
      RC_ASSERT(v.size() <= MAXSIZE);
    });
  }
}

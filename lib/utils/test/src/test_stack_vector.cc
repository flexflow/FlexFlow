#include "doctest.h"
#include "utils/stack_vector.h"
#include <iterator>

using namespace FlexFlow;

TEST_CASE_TEMPLATE("PushBack", T, int, double, char) {
  constexpr std::size_t MAXSIZE = 5;
  using StackVector = stack_vector<T, MAXSIZE>;
  StackVector vector;
  
  vector.push_back(10);
  CHECK_EQ(vector.size(), 1);
  CHECK_EQ(vector[0], 10);

  vector.push_back(20);
  CHECK_EQ(vector.size(), 2);
  CHECK_EQ(vector[0], 10);
  CHECK_EQ(vector[1], 20);
}

TEST_CASE_TEMPLATE("OperatorIndex", T, int, double, char) {
  constexpr std::size_t MAXSIZE = 5;
  using StackVector = stack_vector<T, MAXSIZE>;
  StackVector vector;
  
  vector.push_back(10);
  vector.push_back(20);
  vector.push_back(30);

  CHECK_EQ(vector[0], 10);
  CHECK_EQ(vector[1], 20);
  CHECK_EQ(vector[2], 30);
}

TEST_CASE_TEMPLATE("Size", T, int, double, char) {
  constexpr std::size_t MAXSIZE = 5;
  using StackVector = stack_vector<T, MAXSIZE>;
  StackVector vector;
  
  CHECK_EQ(vector.size(), 0);

  vector.push_back(10);
  CHECK_EQ(vector.size(), 1);

  vector.push_back(20);
  CHECK_EQ(vector.size(), 2);
}

TEST_CASE_TEMPLATE("==", T, int, double, char) {
  constexpr std::size_t MAXSIZE = 5;
  using StackVector = stack_vector<T, MAXSIZE>;
  StackVector vector1, vector2;
 
  vector1.push_back(10);
  vector1.push_back(15);
  vector1.push_back(20);

  vector2.push_back(10);
  vector2.push_back(15);
  vector2.push_back(20);

  CHECK_EQ(vector1, vector2);
}

TEST_CASE_TEMPLATE("EmplaceBack", T, int, double, char) {
  constexpr std::size_t MAXSIZE = 5;
  using StackVector = stack_vector<T, MAXSIZE>;
  StackVector vector;
  
  vector.push_back(10);
  CHECK_EQ(vector.back(), 10);

  vector.push_back(20);
  CHECK_EQ(vector.back(), 20);
}

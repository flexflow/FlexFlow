#include "utils/testing.h"
#include "utils/tuple_extra/prepend.h"

struct type0 { 
  int x; 
  friend bool operator==(type0 const &lhs, type0 const &rhs) {
    return lhs.x == rhs.x;
  }
};
struct type1 { 
  int x; 
  friend bool operator==(type1 const &lhs, type1 const &rhs) {
    return lhs.x == rhs.x;
  }
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tuple_prepend") {
    int head = 5;
    std::tuple<type0, type1, type0> tail = {
      {1},
      {3},
      {4},
    };

    std::tuple<int, type0, type1, type0> correct = {
      head,
      std::get<0>(tail),
      std::get<1>(tail),
      std::get<2>(tail),
    };

    std::tuple<int, type0, type1, type0> result = tuple_prepend(head, tail);
    CHECK_EQ(result, correct);
  }
}

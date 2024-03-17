#include "utils/testing.h"
#include "utils/compile_time_sequence/transform.h"

struct type0 { };
struct type1 { };
struct type2 { };

struct example_function {
  type0 operator()(std::integral_constant<int, 0>) { return type0{}; };
  type1 operator()(std::integral_constant<int, 1>) { return type1{}; };
  type2 operator()(std::integral_constant<int, 2>) { return type2{}; };
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("seq_transform") {
    std::tuple<type2, type0> result = seq_transform(example_function{}, std::declval<seq<2, 0>>());
    std::tuple<type2, type0> correct = std::tuple{type2{}, type0{}};
    CHECK_EQ(result, correct);
  }
}

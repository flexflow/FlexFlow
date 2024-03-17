#include "utils/testing.h"
#include "utils/compile_time_sequence/transform_type.h"
#include <string>

struct type0;
struct type1;
struct type2;

struct example_function {
  type0 operator()(std::integral_constant<int, 0>);
  type1 operator()(std::integral_constant<int, 1>);
  type2 operator()(std::integral_constant<int, 2>);
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("seq_transform_type") {
    CHECK_SAME_TYPE(seq_transform_type<example_function, seq<0, 2>>, type_list<type0, type1>);
  }
}

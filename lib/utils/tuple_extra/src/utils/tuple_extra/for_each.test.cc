#include "utils/tuple_extra/for_each.h"
#include "utils/overload/overload.h"
#include "utils/testing.h"

struct type0 {};
struct type1 {};
struct type2 {};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("for_each(std::tuple<...> const &, F const &)") {
    std::tuple<type0, type1, type2, type1> input = {{}, {}, {}, {}};
    int correct = 1 + 2 + 3 + 2;
    int result = 0;
    auto example_functor = overload{
        [&](type0) { result += 1; },
        [&](type1) { result += 2; },
        [&](type2) { result += 3; },
    };
    for_each(input, example_functor);
    CHECK_EQ(result, correct);
  }
}

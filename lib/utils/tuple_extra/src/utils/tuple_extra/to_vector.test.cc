#include "utils/testing.h"
#include "utils/tuple_extra/to_vector.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("to_vector(std::tuple<...>)") {
    std::tuple<int, int, int> input = {0, 2, 4};
    std::vector<int> correct = { 
      std::get<0>(input),
      std::get<1>(input),
      std::get<2>(input),
    };
    std::vector<int> result = to_vector(input);

    CHECK_EQ(result, correct);
  }
}

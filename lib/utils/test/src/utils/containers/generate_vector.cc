#include "utils/containers/generate_vector.h"
#include "test/utils/doctest/fmt/vector.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("generate_vector") {
    SUBCASE("empty vector") {
      std::vector<int> result = generate_vector(0_n, [](nonnegative_int) -> int {
        PANIC("lambda should not be called");
      });
      std::vector<int> correct = {};

      CHECK(result == correct);
    }
    SUBCASE("non-empty vector") {
      std::vector<int> result = generate_vector(5_n, [](nonnegative_int idx) {
        int i = idx.unwrap_nonnegative();
        return i * i;

      });
      std::vector<int> correct = {0, 1, 4, 9, 16};

      CHECK(result == correct);
    }
  }
}

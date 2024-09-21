#include "utils/containers/repeat.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("repeat") {
    int x = 0;
    std::vector<int> result = repeat(3, [&]() {
      int result = x;
      x += 2;
      return result;
    });

    std::vector<int> correct = {0, 2, 4};

    CHECK(result == correct);
  }
}

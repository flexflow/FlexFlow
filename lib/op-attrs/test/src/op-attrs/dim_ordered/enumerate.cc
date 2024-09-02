#include "op-attrs/dim_ordered/enumerate.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/map.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("enumerate(FFOrdered<T>)") {
    FFOrdered<std::string> input = {"zero", "one", "two"};

    std::map<ff_dim_t, std::string> result = enumerate(input);
    std::map<ff_dim_t, std::string> correct = {
        {ff_dim_t{0}, "zero"},
        {ff_dim_t{1}, "one"},
        {ff_dim_t{2}, "two"},
    };

    CHECK(result == correct);
  }
}

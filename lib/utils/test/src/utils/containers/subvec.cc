#include "utils/containers/subvec.h"
#include <doctest/doctest.h>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Testing subvec function") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto subvec_v = subvec(v, std::optional<int>(1), std::optional<int>(4));

    CHECK(subvec_v == std::vector<int>({2, 3, 4}));

    auto subvec_v2 = subvec(v, std::nullopt, std::optional<int>(3));
    CHECK(subvec_v2 == std::vector<int>({1, 2, 3}));
  }
}

#include "utils/containers/subvec.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("subvec") {
    std::vector<int> v = {1, 2, 3, 4, 5};

    SUBCASE("Basic subvector") {
      auto result = subvec(v, 1, 4);
      std::vector<int> correct = {2, 3, 4};
      CHECK(result == correct);
    }

    SUBCASE("From beginning to index") {
      auto result = subvec(v, std::nullopt, 3);
      std::vector<int> correct = {1, 2, 3};
      CHECK(result == correct);
    }

    SUBCASE("From index to end") {
      auto result = subvec(v, 2, std::nullopt);
      std::vector<int> correct = {3, 4, 5};
      CHECK(result == correct);
    }

    SUBCASE("All of the vector") {
      auto result = subvec(v, std::nullopt, std::nullopt);
      std::vector<int> correct = {1, 2, 3, 4, 5};
      CHECK(result == correct);
    }

    SUBCASE("Start greater than end") {
      auto result = subvec(v, 3, 1);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("Start equal to end") {
      auto result = subvec(v, 3, 3);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("Negative indices") {
      auto result = subvec(v, -3, -1);
      std::vector<int> correct = {3, 4};
      CHECK(result == correct);
    }

    SUBCASE("Out of bounds index from above") {
      CHECK_THROWS(subvec(v, 2, 100));
    }

    SUBCASE("Out of bounds index from below") {
      CHECK_THROWS(subvec(v, -100, 2));
    }
  }
}

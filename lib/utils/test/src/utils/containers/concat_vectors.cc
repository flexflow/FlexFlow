#include "utils/containers/concat_vectors.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("concat_vectors") {
    SUBCASE("two vectors") {
      std::vector<int> prefix = {1, 2};
      std::vector<int> postfix = {3, 4};

      std::vector<int> result = concat_vectors(prefix, postfix);
      std::vector<int> correct = {1, 2, 3, 4};

      CHECK(result == correct);
    }

    SUBCASE("vector of vectors") {
      std::vector<std::vector<int>> vecs = {{1, 2}, {3}, {4, 5}};

      std::vector<int> result = concat_vectors(vecs);
      std::vector<int> correct = {1, 2, 3, 4, 5};

      CHECK(result == correct);
    }

    SUBCASE("empty vector of vectors") {
      std::vector<std::vector<int>> vecs = {};

      std::vector<int> result = concat_vectors(vecs);
      std::vector<int> correct = {};

      CHECK(result == correct);
    }
  }
}

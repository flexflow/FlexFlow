#include "utils/containers/extend.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("extend(std::vector &, C)") {
    std::vector<int> result = {1, 2, 3};
    std::vector<int> rhs = {3, 4, 5};
    extend(result, rhs);

    std::vector<int> correct = {1, 2, 3, 3, 4, 5};

    CHECK(result == correct);
  }

  TEST_CASE("extend(std::vector<T> &, std::optional<E>)") {
    std::vector<int> result = {1, 2, 3};

    SUBCASE("rhs has value") {
      std::optional<int> rhs = 2;
      extend(result, rhs);
      std::vector<int> correct = {1, 2, 3, 2};
      CHECK(result == correct);
    }

    SUBCASE("rhs is nullopt") {
      std::optional<int> rhs = std::nullopt;
      extend(result, rhs);
      std::vector<int> correct = {1, 2, 3};
      CHECK(result == correct);
    }
  }

  TEST_CASE("extend(std::unordered_set<T> &, C)") {
    std::unordered_set<int> result = {1, 2, 3};
    std::vector<int> rhs = {3, 3, 4, 5};
    extend(result, rhs);

    std::unordered_set<int> correct = {1, 2, 3, 4, 5};

    CHECK(result == correct);
  }

  TEST_CASE("extend(std::unordered_set<T> &, std::optional<E>)") {
    std::unordered_set<int> result = {1, 2, 3};

    SUBCASE("rhs has value") {
      std::optional<int> rhs = 5;
      extend(result, rhs);
      std::unordered_set<int> correct = {1, 2, 3, 5};
      CHECK(result == correct);
    }

    SUBCASE("rhs is nullopt") {
      std::optional<int> rhs = std::nullopt;
      extend(result, rhs);
      std::unordered_set<int> correct = {1, 2, 3};
      CHECK(result == correct);
    }
  }
}

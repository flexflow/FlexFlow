#include "utils/containers/require_all_same1.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/multiset.h"
#include "utils/fmt/optional.h"
#include "utils/fmt/set.h"
#include "utils/fmt/unordered_multiset.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>
#include <optional>
#include <set>
#include <unordered_set>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("require_all_same1(T)",
                     T,
                     std::vector<int>,
                     std::unordered_set<int>,
                     std::unordered_multiset<int>,
                     std::set<int>,
                     std::multiset<int>) {
    SUBCASE("input is empty") {
      T input = {};

      std::optional<int> result =
          optional_from_expected(require_all_same1(input));
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input elements are all the same") {
      T input = {1, 1, 1};

      tl::expected<int, std::string> result = require_all_same1(input);
      tl::expected<int, std::string> correct = 1;

      CHECK(result == correct);
    }

    SUBCASE("input elements are not all the same") {
      T input = {1, 1, 2, 1};

      std::optional<int> result =
          optional_from_expected(require_all_same1(input));
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
